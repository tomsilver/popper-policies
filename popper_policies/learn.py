"""Learn policies for PDDL domains using Popper (ILP system)."""

import logging
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List, Optional, Set, Tuple

from popper.loop import learn_solution
from popper.util import Settings as PopperSettings
from popper.util import order_prog, order_rule

from popper_policies import utils
from popper_policies.structs import LiftedDecisionListPolicy, \
    LiftedDecisionListRule, Plan, StateGoalAction, Task


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

    # Convert the plans into state-goal-action triplets.
    demo_state_goal_actions = []
    for task, plan in zip(tasks, plan_strs):
        for state, goal, action in utils.plan_to_trajectory(task, plan):
            demo_state_goal_actions.append((state, goal, action))

    # The background knowledge is constant across all actions.
    bk_str = _create_background_knowledge(demo_state_goal_actions)
    logging.debug(f"Created background string:\n{bk_str}")

    programs = []

    for action in actions:
        logging.info(f"Learning rules for action: {action}")

        # Create temporary directory to store the files.
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            # Create the bias file.
            # NOTE: Prolog complains if we introduce an unused predicate, so
            # just collect seen predicates from the problems themselves.
            bias_str = _create_bias(tasks, action)
            logging.debug(f"Created bias string:\n{bias_str}")
            bias_file = temp_dir_path / "bias.pl"
            with open(bias_file, "w", encoding="utf-8") as f:
                f.write(bias_str)

            # Create the background knowledge (bk) file.
            bk_file = temp_dir_path / "bk.pl"
            with open(bk_file, "w", encoding="utf-8") as f:
                f.write(bk_str)

            # Create the examples (exs) file.
            examples_str = _create_examples(demo_state_goal_actions, tasks,
                                            action)
            logging.debug(f"Created examples string:\n{examples_str}")
            examples_file = temp_dir_path / "exs.pl"
            with open(examples_file, "w", encoding="utf-8") as f:
                f.write(examples_str)

            # Call popper.
            logging.debug(f"Calling popper.")
            settings = PopperSettings(kbpath=temp_dir)
            # TODO: debug potential interference between successive calls.
            # https://github.com/logic-and-learning-lab/Popper/issues/62
            prog, _, _ = learn_solution(settings)
            assert prog is not None
            programs.append(prog)

    return _popper_programs_to_policy(programs)


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


def _atom_str_to_prolog_str(atom_str: str,
                            example_id: int,
                            wrapper: Optional[str] = None) -> str:
    """Reformat atom string to include the example id, and remove spaces."""
    assert atom_str.startswith("(")
    name, remainder = atom_str[1:].split(" ", 1)
    s = f"{name}({remainder}"
    # Remove spaces.
    s = s.replace(" ", ",")
    # Add task id.
    assert s.endswith(")")
    s = f"{s[:-1]},{example_id})"
    if wrapper is not None:
        s = f"{wrapper}({s})"
    s += "."
    return s


def _create_background_knowledge(
        demo_state_goal_actions: List[StateGoalAction]) -> str:
    """Returns the content of a Popper background knowledge file."""
    # Prolog complains if the file is not organized by predicate.
    pred_to_strs: DefaultDict[Tuple[str, int], Set[str]] = defaultdict(set)

    for example_id, (state, goal, _) in enumerate(demo_state_goal_actions):
        for atom in state:
            name = atom.name
            arity = len(atom.signature)
            pred = (name, arity)
            pred_str = utils.pred_to_str(atom)
            prolog_str = _atom_str_to_prolog_str(pred_str, example_id)
            pred_to_strs[pred].add(prolog_str)

        for atom in goal:
            name = atom.name
            goal_name = "goal_" + atom.name
            arity = len(atom.signature)
            pred = (goal_name, arity)
            pred_str = utils.pred_to_str(atom).replace(name, goal_name, 1)
            prolog_str = _atom_str_to_prolog_str(pred_str, example_id)
            pred_to_strs[pred].add(prolog_str)

    # Finalize background knowledge.
    bk_str = ""
    for pred in sorted(pred_to_strs):
        bk_str += "\n".join(sorted(pred_to_strs[pred]))
        bk_str += "\n"
    return bk_str


def _create_examples(demo_state_goal_actions: List[StateGoalAction],
                     tasks: List[Task], action: Tuple[str, int]) -> str:
    """Returns the content of a Popper examples file.

    Currently makes the (often incorrect!) assumption that there is only
    one "good" action for each (state, goal), i.e., the demonstrated
    action. All other possible actions are treated as negative examples.
    """
    # Generate all possible actions for the sake of negative examples.
    # We may want to subsample in the future because this could get very big.
    all_possible_actions = set()
    for task in tasks:
        for op in task.pyperplan_task.operators:
            if not op.name.startswith(f"({action[0]}"):
                continue
            all_possible_actions.add(op.name)

    pos_strs = set()
    neg_strs = set()
    for ex_id, (_, _, action) in enumerate(demo_state_goal_actions):
        for other_action in all_possible_actions:
            # Positive example
            if action == other_action:
                wrapper = "pos"
                destination = pos_strs
            # Negative example
            else:
                wrapper = "neg"
                destination = neg_strs
            prolog_str = _atom_str_to_prolog_str(other_action,
                                                 ex_id,
                                                 wrapper=wrapper)
            destination.add(prolog_str)

    pos_str = "\n".join(sorted(pos_strs))
    neg_str = "\n".join(sorted(neg_strs))

    return f"""% Positive examples
{pos_str}

% Negative examples
{neg_str}
"""


def _popper_programs_to_policy(popper_programs):
    policy_rules = []
    for prog in popper_programs:
        for rule in order_prog(prog):
            act_literal, body = order_rule(rule)
            # Parse action.
            assert act_literal
            action_str = str(act_literal.predicate)
            action_arg_strs = [str(a) for a in act_literal.arguments]
            action = (action_str, action_arg_strs)
            # Parse conditions.
            conditions = []
            for cond in body:
                pred_str = str(cond.predicate)
                pred_arg_strs = [str(a) for a in cond.arguments]
                conditions.append(pred_str, pred_arg_strs)
            rule = LiftedDecisionListRule(conditions, action)
            policy_rules.append(rule)
    return LiftedDecisionListPolicy(policy_rules)
