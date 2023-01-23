"""Learn policies for PDDL domains using Popper (ILP system)."""

import logging
import tempfile
from pathlib import Path
from typing import List, Set, Tuple

from popper_policies.structs import Plan, Task


def learn_policy(domain_str: str, problem_strs: List[str],
                 plan_strs: List[Plan]) -> str:
    """Learn a goal-conditioned policy using Popper."""
    # Parse the PDDL.
    tasks = [Task(domain_str, problem_str) for problem_str in problem_strs]

    # Collect all actions seen in the plans; learn one program per action.
    # Actions are recorded with their names and arities.
    action_set: Set[Tuple[str, int]] = set()
    for plan in plan_strs:
        for ground_action in plan:
            assert ground_action.startswith("(")
            action, remainder = ground_action[1:].split(" ", 1)
            arity = len(remainder.split(" "))
            action_set.add((action, arity))
    actions = sorted(action_set)
    logging.info(f"Found actions in plans: {actions}")

    for action in actions:
        logging.info(f"Learning rules for action: {action}")

        # Create temporary directory to store the files.
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            # Create the bias file.
            # NOTE: Prolog complains if we introduce an unused predicate, so
            # just collect seen predicates from the problems themselves.
            bias_str = _create_bias(tasks, action)
            bias_file = temp_dir_path / "bias.pl"
            with open(bias_file, "w", encoding="utf-8") as f:
                f.write(bias_str)

            # Create the background knowledge (bk) file.
            import ipdb; ipdb.set_trace()

            # Create the examples (exs) file.
            import ipdb; ipdb.set_trace()


def _create_bias(tasks: List[Task], action: Tuple[str, int]) -> str:
    """Returns the content of a Popper bias file."""
    action_name, action_arity = action

    # Collect all predicates and goal predicates with their names and arities.
    predicates: Set[Tuple[str, int]] = set()
    goal_predicates: Set[Tuple[str, int]] = set()
    for task in tasks:
        for atom in task.problem.initial_state:
            name = atom.name
            arity = len(atom.signature)
            predicates.add((name, arity))
            assert not name.startswith("goal_")
        for atom in task.problem.goal:
            name = atom.name
            arity = len(atom.signature)
            goal_predicates.add((name, arity))

    # Create predicate and goal predicate strings.
    pred_str = "\n".join(f"body_pred({name},{arity+1})."
                         for name, arity in sorted(predicates))
    goal_pred_str = "\n".join(f"body_pred(goal_{name},{arity+1})."
                              for name, arity in sorted(goal_predicates))

    return f"""% Predicates
{pred_str}

% Goal predicates
{goal_pred_str}

% Action
head_pred({action_name},{action_arity+1}).
"""
