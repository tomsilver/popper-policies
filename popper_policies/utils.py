"""Utilities."""

import functools
import hashlib
import itertools
import logging
import os
import re
import subprocess
import tempfile
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Collection, Dict, FrozenSet, Iterator, List, \
    Optional, Sequence, Set, Tuple

from pyperplan.planner import HEURISTICS, SEARCHES, search_plan

from popper_policies.flags import FLAGS
from popper_policies.structs import LDLRule, LiftedDecisionList, Plan, \
    PyperplanAction, PyperplanEffect, PyperplanObject, PyperplanOperator, \
    PyperplanPredicate, PyperplanType, StateGoalAction, Task, TaskMetrics, \
    _GroundLDLRule

# Global constants.
_DIR = Path(__file__).parent
PDDL_DIR = _DIR / "envs" / "assets" / "pddl"


@functools.lru_cache(maxsize=None)
def get_git_commit_hash() -> str:
    """Return the hash of the current git commit."""
    out = subprocess.check_output(["git", "rev-parse", "HEAD"])
    return out.decode("ascii").strip()


def get_pddl_from_url(url: str, cache_dir: Path = PDDL_DIR) -> str:
    """Download a PDDL file from a given URL.

    If the file already exists in the cache_dir, load instead.

    Note that this assumes the PDDL won't change at the URL.
    """
    sanitized_url = "".join(x for x in url if x.isalnum())
    file_name = f"cached-pddl-{sanitized_url}"
    file_path = cache_dir / file_name
    # Download if doesn't already exist.
    if not os.path.exists(file_path):
        logging.info(f"Cache not found for {url}, downloading.")
        try:
            with urllib.request.urlopen(url) as f:
                pddl = f.read().decode('utf-8').lower()
        except urllib.error.HTTPError:
            raise ValueError(f"PDDL file not found at {url}.")
        if "(:action" not in pddl and "(:init" not in pddl:
            raise ValueError(f"PDDL file not found at {url}.")
        # Add a note at the top of the file about when this was downloaded.
        today = date.today().strftime("%B %d, %Y")
        note = f"; Downloaded {today} from {url}\n"
        pddl = note + pddl
        # Cache.
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(pddl)
    with open(file_path, "r", encoding="utf-8") as f:
        pddl = f.read()
    return pddl


def run_planning(
        task: Task,
        planner: str = "pyperplan") -> Tuple[Optional[Plan], TaskMetrics]:
    """Find a plan."""
    if planner == "pyperplan":
        return run_pyperplan_planning(task)
    if planner == "fastdownward":  # pragma: no cover
        return run_fastdownward_planning(task)
    if planner == "fastdownward-hff-gbfs":  # pragma: no cover
        return run_fastdownward_planning(task,
                                         alias=None,
                                         search="eager_greedy([ff()])")
    raise NotImplementedError(f"Unrecognized planner: {planner}")


def run_pyperplan_planning(
    task: Task,
    heuristic: str = "lmcut",
    search: str = "astar",
) -> Tuple[Optional[Plan], TaskMetrics]:
    """Find a plan with pyperplan."""
    search_fn = SEARCHES[search]
    heuristic_fn = HEURISTICS[heuristic]
    # Quiet the pyperplan logging.
    logging.disable(logging.ERROR)
    pyperplan_plan = search_plan(
        task.domain_file,
        task.problem_file,
        search_fn,
        heuristic_fn,
    )
    logging.disable(logging.NOTSET)
    metrics: TaskMetrics = {
    }  # currently not collecting metrics from pyperplan
    if pyperplan_plan is None:
        return None, metrics
    return [a.name for a in pyperplan_plan], metrics


def run_fastdownward_planning(
    task: Task,
    alias: Optional[str] = "lama-first",
    search: Optional[str] = None,
) -> Tuple[Optional[Plan], TaskMetrics]:  # pragma: no cover
    """Find a plan with fast downward.

    Usage: Build and compile the Fast Downward planner, then set the environment
    variable FD_EXEC_PATH to point to the `downward` directory. For example:
    1) git clone https://github.com/aibasel/downward.git
    2) cd downward && ./build.py
    3) export FD_EXEC_PATH="<your absolute path here>/downward"
    """
    # Specify either a search flag or an alias.
    assert (search is None) + (alias is None) == 1
    # The SAS file isn't actually used, but it's important that we give it a
    # name, because otherwise Fast Downward uses a fixed default name, which
    # will cause issues if you run multiple processes simultaneously.
    sas_file = tempfile.NamedTemporaryFile(delete=False).name
    # Run Fast Downward followed by cleanup. Capture the output.
    assert "FD_EXEC_PATH" in os.environ, \
        "Please follow the instructions in the docstring of this method!"
    if alias is not None:
        alias_flag = f"--alias {alias}"
    else:
        alias_flag = ""
    if search is not None:
        search_flag = f"--search '{search}'"
    else:
        search_flag = ""
    fd_exec_path = os.environ["FD_EXEC_PATH"]
    exec_str = os.path.join(fd_exec_path, "fast-downward.py")
    cmd_str = (f'"{exec_str}" {alias_flag} '
               f'--sas-file {sas_file} '
               f'"{task.domain_file}" "{task.problem_file}" '
               f'{search_flag}')
    output = subprocess.getoutput(cmd_str)
    cleanup_cmd_str = f"{exec_str} --cleanup"
    subprocess.getoutput(cleanup_cmd_str)
    # Parse and log metrics.
    num_nodes_expanded = re.findall(r"Expanded (\d+) state", output)[0]
    num_nodes_created = re.findall(r"Evaluated (\d+) state", output)[0]
    metrics = {
        "nodes_expanded": float(num_nodes_expanded),
        "nodes_created": float(num_nodes_created)
    }
    # Extract the plan from the output, if one exists.
    if "Solution found!" not in output:
        return None, metrics
    if "Plan length: 0 step" in output:
        # Handle the special case where the plan is found to be trivial.
        return [], metrics
    plan_str = re.findall(r"(.+) \(\d+?\)", output)
    assert plan_str  # already handled empty plan case, so something went wrong
    plan = [f"({a})" for a in plan_str]
    return plan, metrics


def pred_to_str(pred: PyperplanPredicate) -> str:
    """Create a string representation of a Pyperplan predicate (atom)."""
    if not pred.signature:
        return f"({pred.name})"
    arg_str = " ".join(str(o) for o, _ in pred.signature)
    return f"({pred.name} {arg_str})"


def str_to_pred(
        s: str, pred_library: Dict[str,
                                   PyperplanPredicate]) -> PyperplanPredicate:
    """Create a Pyperplan predicate (atom) from a string representation."""
    assert s.startswith("(")
    assert s.endswith(")")
    s = s[1:-1]
    if " " in s:
        name, remainder = s.split(" ", 1)
        args = remainder.split(" ")
    else:
        name = s
        args = []
    reference_pred = pred_library[name]
    signature = reference_pred.signature
    types = [t for _, t in signature]
    assert len(args) == len(types)
    new_signature = list(zip(args, types))
    return PyperplanPredicate(name, new_signature)


def pred_to_type_names(pred: PyperplanPredicate) -> Tuple[str, ...]:
    """Extract name names from predicate (atom)."""
    names: List[str] = []
    for _, t in pred.signature:
        if isinstance(t, (list, tuple)):
            names.append(t[0].name)
        else:
            names.append(t.name)
    return tuple(names)


def get_objects_str(task: Task, include_constants: bool = False) -> str:
    """Returns a PDDL encoding of the objects in the task."""
    # Create the objects string.
    type_to_objs: Dict[PyperplanType, List[PyperplanObject]] = {
        t: []
        for t in task.domain.types.values()
    }
    for obj in sorted(task.problem.objects):
        if obj in task.domain.constants and not include_constants:
            continue
        obj_type = task.problem.objects[obj]
        type_to_objs[obj_type].append(obj)
    # Construct the object list for the prompt.
    objects_strs: List[str] = []
    for typ, objs in type_to_objs.items():
        if not objs:
            continue
        typ_str = " ".join(objs) + " - " + str(typ)
        objects_strs.append(typ_str)
    return "\n  ".join(objects_strs)


def get_init_strs(task: Task) -> List[str]:
    """Returns the init strings of a PDDL task."""
    init_strs = [pred_to_str(p) for p in task.problem.initial_state]
    return init_strs


def get_init_str(task: Task) -> str:
    """Returns the init string of a PDDL task."""
    init_strs = get_init_strs(task)
    return "\n".join(init_strs)


def get_goal_strs(task: Task) -> List[str]:
    """Returns the goal strings of a PDDL task."""
    return [pred_to_str(p) for p in task.problem.goal]


def get_goal_str(task: Task) -> str:
    """Returns the goal string of a PDDL task."""
    goal_strs = get_goal_strs(task)
    goal_str = "\n".join(goal_strs)
    return goal_str


def str_to_identifier(x: str) -> str:
    """Convert a string to a small string with negligible collision probability
    and where the smaller string can be used to identifier the larger string in
    file names.

    Importantly, this function is deterministic between runs and between
    platforms, unlike python's built-in hash function.
    References:
        https://stackoverflow.com/questions/45015180
        https://stackoverflow.com/questions/5297448
    """
    return hashlib.md5(x.encode('utf-8')).hexdigest()


def is_subtype(type1: PyperplanType, type2: PyperplanType) -> bool:
    """Checks whether type1 inherits from type2."""
    while type1 is not None:
        if type1 == type2:
            return True
        type1 = type1.parent
    return False


def reset_flags(args: Dict[str, Any], default_seed: int = 123) -> None:
    """Resets FLAGS for use in unit tests.

    Unless seed is specified, we use a default for testing.
    """
    FLAGS.__dict__.clear()
    FLAGS.__dict__.update(args)
    if "seed" not in FLAGS:
        FLAGS.__dict__["seed"] = default_seed


def action_to_task_operator(task: Task, action: str) -> PyperplanOperator:
    """Look up operator for action and raise ValueError if not found."""
    pyperplan_task = task.pyperplan_task
    for op in pyperplan_task.operators:
        if op.name == action:
            action_op = op
            break
    else:  # pragma: no cover
        raise ValueError(f"Invalid action for task: {action}")
    return action_op


def action_is_valid_for_task(task: Task, action: str) -> bool:
    """Check whether the action is valid in the task initial state."""
    pyperplan_task = task.pyperplan_task
    current_facts = pyperplan_task.initial_state
    try:
        action_op = action_to_task_operator(task, action)
    except ValueError as e:
        assert "Invalid action" in str(e)
        return False
    return action_op.applicable(current_facts)


def advance_task(task: Task, action: str) -> Task:
    """Create a new task with a new initial state."""
    action_op = action_to_task_operator(task, action)
    objects_str = get_objects_str(task, include_constants=False)
    init_strs = set(get_init_strs(task))
    init_strs = (init_strs - action_op.del_effects) | action_op.add_effects
    init_str = "\n  ".join(sorted(init_strs))
    goal_str = get_goal_str(task)
    new_problem_str = f"""(define (problem synthetic-problem)
   (:domain {task.domain.name})
  (:objects
  {objects_str}
  )
  (:init
  {init_str}
  )
  (:goal (and
  {goal_str})
  )
)"""
    new_task = Task(task.domain_str, new_problem_str)
    return new_task


def plan_to_trajectory(task: Task, plan: Plan) -> Iterator[StateGoalAction]:
    """Iterate state-goal-action triplets by running the plan in the task."""
    goal = set(task.problem.goal)
    objects = dict(task.problem.objects)
    for action in plan:
        yield (set(task.problem.initial_state), goal, objects, action)
        assert action_is_valid_for_task(task, action)
        task = advance_task(task, action)


def all_ground_ldl_rules(
        rule: LDLRule,
        objects: Set[Tuple[str, PyperplanType]]) -> List[_GroundLDLRule]:
    """Get all possible groundings of the given rule with the given objects."""
    return _cached_all_ground_ldl_rules(rule, frozenset(objects))


def get_object_combinations(
        entities: Collection[Tuple[str, PyperplanType]],
        types: Sequence[List[PyperplanType]]) -> Iterator[List[str]]:
    """Get all combinations of entities satisfying the given types sequence."""
    sorted_entities = sorted(entities)
    choices = []
    for vt in types:
        this_choices = []
        for (ent, et) in sorted_entities:
            if any(et.name == t.name for t in vt):
                this_choices.append(ent)
        choices.append(this_choices)
    for choice in itertools.product(*choices):
        yield list(choice)


@functools.lru_cache(maxsize=None)
def _cached_all_ground_ldl_rules(
    rule: LDLRule,
    frozen_objects: FrozenSet[Tuple[str,
                                    PyperplanType]]) -> List[_GroundLDLRule]:
    """Helper for all_ground_ldl_rules() that caches the outputs."""
    ground_rules = []
    types = [t for _, t in rule.parameters]
    for choice in get_object_combinations(frozen_objects, types):
        ground_rule = rule.ground(tuple(choice))
        ground_rules.append(ground_rule)
    return ground_rules


def query_ldl(ldl: LiftedDecisionList, atoms: Set[str],
              objects: Set[Tuple[str, PyperplanType]],
              goal: Set[str]) -> Optional[str]:
    """Queries a lifted decision list representing a goal-conditioned policy.

    Given an abstract state and goal, the rules are grounded in order. The
    first applicable ground rule is used to return a ground action.

    If no rule is applicable, returns None.
    """
    for rule in ldl.rules:
        for ground_rule in all_ground_ldl_rules(rule, objects):
            if ground_rule.pos_state_preconditions.issubset(atoms) and \
               not ground_rule.neg_state_preconditions & atoms and \
               ground_rule.goal_preconditions.issubset(goal):
                return ground_rule.ground_operator
    return None


@functools.singledispatch
def apply_substitution(target, sub: Dict[str, str]):
    """Apply a substitution to a pyperplan struct."""
    raise NotImplementedError("See below.")


@apply_substitution.register
def _(target: PyperplanPredicate, sub: Dict[str, str]) -> PyperplanPredicate:
    """Apply a substitution to a predicate."""
    orig_signature = target.signature
    new_signature = [(sub[p], t) for p, t in orig_signature]
    return PyperplanPredicate(target.name, new_signature)


@apply_substitution.register
def _(target: PyperplanAction, sub: Dict[str, str]) -> PyperplanAction:
    """Apply a substitution to an operator."""
    orig_signature = target.signature
    new_signature = [(sub[p], t) for p, t in orig_signature]
    new_preconds = {apply_substitution(e, sub) for e in target.precondition}
    new_effects = PyperplanEffect()
    old_effects = target.effect
    new_effects.addlist = {
        apply_substitution(e, sub)
        for e in old_effects.addlist
    }
    new_effects.dellist = {
        apply_substitution(e, sub)
        for e in old_effects.dellist
    }
    return PyperplanAction(target.name, new_signature, new_preconds,
                           new_effects)


@apply_substitution.register
def _(target: LDLRule, sub: Dict[str, str]) -> PyperplanPredicate:
    """Apply a substitution to a LDL rule."""
    new_parameters = [(sub[old], t) for old, t in target.parameters]
    new_pos_state_preconditions = {
        apply_substitution(a, sub)
        for a in target.pos_state_preconditions
    }
    new_neg_state_preconditions = {
        apply_substitution(a, sub)
        for a in target.neg_state_preconditions
    }
    new_goal_preconditions = {
        apply_substitution(a, sub)
        for a in target.goal_preconditions
    }
    new_operator = apply_substitution(target.operator, sub)

    return LDLRule(target.name, new_parameters, new_pos_state_preconditions,
                   new_neg_state_preconditions, new_goal_preconditions,
                   new_operator)
