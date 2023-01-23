"""Data structures."""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

from pyperplan.grounding import ground as pyperplan_ground
from pyperplan.pddl.parser import Parser
from pyperplan.pddl.pddl import Action as PyperplanAction
from pyperplan.pddl.pddl import Domain as PyperplanDomain
from pyperplan.pddl.pddl import \
    Effect as PyperplanEffect  # pylint: disable=unused-import
from pyperplan.pddl.pddl import \
    Predicate as PyperplanPredicate  # pylint: disable=unused-import
from pyperplan.pddl.pddl import Problem as PyperplanProblem
from pyperplan.pddl.pddl import \
    Type as PyperplanType  # pylint: disable=unused-import
from pyperplan.task import \
    Operator as PyperplanOperator  # pylint: disable=unused-import
from pyperplan.task import Task as PyperplanTask

PyperplanObject = str


@dataclass(frozen=True)
class Task:
    """A task is a PDDL domain str and problem str."""
    domain_str: str
    problem_str: str

    @cached_property
    def domain_file(self) -> Path:
        """A file that contains the domain str."""
        filename = tempfile.NamedTemporaryFile(delete=False).name
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.domain_str)
        return Path(filename)

    @cached_property
    def problem_file(self) -> Path:
        """A file that contains the problem str."""
        filename = tempfile.NamedTemporaryFile(delete=False).name
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.problem_str)
        return Path(filename)

    @cached_property
    def _parser(self) -> Parser:
        return Parser(self.domain_file, self.problem_file)

    @cached_property
    def domain(self) -> PyperplanDomain:
        """The parsed PDDL domain for this task."""
        return self._parser.parse_domain()

    @cached_property
    def problem(self) -> PyperplanProblem:
        """The parsed PDDL problem for this task."""
        return self._parser.parse_problem(self.domain)

    @cached_property
    def size(self) -> int:
        """A crude measure of task complexity."""
        prob = self.problem
        return len(prob.objects) + len(prob.initial_state) + len(prob.goal)

    @cached_property
    def pyperplan_task(self) -> PyperplanTask:
        """The pyperplan task for this task."""
        logging.disable(logging.ERROR)
        pyperplan_task = pyperplan_ground(self.problem)
        logging.disable(logging.NOTSET)
        return pyperplan_task


# A plan is currently just a list of strings, where each string is one ground
# operator, e.g., (unstack a b). We may change this later.
Plan = List[str]
StateGoalAction = Tuple[Set[PyperplanPredicate], Set[PyperplanPredicate], str]

# Metrics are saved during evaluation.
TaskMetrics = Dict[str, Any]
# Maps a task string identifier to task metrics.
Metrics = Dict[str, TaskMetrics]


@dataclass(frozen=True, repr=False, eq=False)
class LDLRule:
    """A lifted decision list rule."""
    name: str
    parameters: Sequence[Tuple[str, PyperplanType]]
    state_preconditions: Set[PyperplanPredicate]
    goal_preconditions: Set[PyperplanPredicate]
    operator: PyperplanAction

    @lru_cache(maxsize=None)
    def ground(self, objects: Tuple[str, ...]) -> _GroundLDLRule:
        """Ground into a _GroundLDLRule, given objects.

        Insist that objects are tuple for hashing in cache.
        """
        assert isinstance(objects, tuple)
        assert len(objects) == len(self.parameters)
        sub = dict(zip(self.parameters, objects))
        pre_s = {_ground_atom(atom, sub) for atom in self.state_preconditions}
        pre_g = {_ground_atom(atom, sub) for atom in self.goal_preconditions}
        op_objects = tuple(sub[v] for v in self.operator.parameters)
        ground_op = _ground_operator(self.operator, op_objects)
        return _GroundLDLRule(self, list(objects), pre_s, pre_g, ground_op)

    @cached_property
    def _str(self) -> str:
        parameter_str = "(" + " ".join(
            [f"{p} - {t[0].name}" for p, t in self.parameters]) + ")"

        def _atom_to_str(atom: PyperplanPredicate) -> str:
            args_str = " ".join([v for v, _ in atom.signature])
            return f"({atom.name} {args_str})"

        inner_preconditions_strs = [
            _atom_to_str(a) for a in sorted(self.state_preconditions)
        ]
        preconditions_str = " ".join(inner_preconditions_strs)
        if len(inner_preconditions_strs) > 1:
            preconditions_str = "(and " + preconditions_str + ")"
        elif not inner_preconditions_strs:
            preconditions_str = "()"
        goals_strs = [_atom_to_str(a) for a in sorted(self.goal_preconditions)]
        goals_str = " ".join(goals_strs)
        if len(goals_strs) > 1:
            goals_str = "(and " + goals_str + ")"
        elif not goals_strs:
            goals_str = "()"
        action_param_str = " ".join([v for v, _ in self.operator.signature])
        action_str = f"({self.operator.name} {action_param_str})"
        return f"""(:rule {self.name}
    :parameters {parameter_str}
    :preconditions {preconditions_str}
    :goals {goals_str}
    :action {action_str}
  )"""

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, LDLRule)
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, LDLRule)
        return str(self) < str(other)

    def __gt__(self, other: object) -> bool:
        assert isinstance(other, LDLRule)
        return str(self) > str(other)


@dataclass(frozen=True, repr=False, eq=False)
class _GroundLDLRule:
    """A ground LDL rule is an LDLRule + objects.

    Should not be instantiated externally.
    """
    parent: LDLRule
    objects: Sequence[str]
    state_preconditions: Set[PyperplanPredicate]
    goal_preconditions: Set[PyperplanPredicate]
    ground_operator: PyperplanAction

    @cached_property
    def _str(self) -> str:
        op_obj_str = ", ".join([str(o) for o in self.ground_operator.objects])
        return f"""GroundLDLRule-{self.name}:
    Parameters: {self.objects}
    State Pre: {sorted(self.state_preconditions, key=str)}
    Goal Pre: {sorted(self.goal_preconditions, key=str)}
    Operator: {self.ground_operator.name}({op_obj_str})"""

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    @property
    def name(self) -> str:
        """Name of this ground LDL rule."""
        return self.parent.name

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, _GroundLDLRule)
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, _GroundLDLRule)
        return str(self) < str(other)

    def __gt__(self, other: object) -> bool:
        assert isinstance(other, _GroundLDLRule)
        return str(self) > str(other)


@dataclass(frozen=True)
class LiftedDecisionList:
    """A goal-conditioned policy from abstract states to ground operators
    implemented with a lifted decision list.

    The logic described above is implemented in utils.query_ldl().
    """
    rules: Sequence[LDLRule]

    @cached_property
    def _hash(self) -> int:
        return hash(tuple(self.rules))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, LiftedDecisionList)
        if len(self.rules) != len(other.rules):
            return False
        return all(r1 == r2 for r1, r2 in zip(self.rules, other.rules))

    def __str__(self) -> str:
        rule_str = "\n  ".join(str(r) for r in self.rules)
        return f"""(define (policy)\n  {rule_str}\n)"""
