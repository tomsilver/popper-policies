"""Learn policies for PDDL domains using Popper (ILP system)."""

import logging
import multiprocessing
import tempfile
from collections import defaultdict
from multiprocessing import Process
from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

from popper.loop import learn_solution
from popper.util import Settings as PopperSettings
from popper.util import order_prog, order_rule

from popper_policies import utils
from popper_policies.structs import LDLRule, LiftedDecisionList, Plan, \
    PyperplanDomain, PyperplanPredicate, StateGoalAction, Task

# Predicate substitutions, action substitutions.
_DomainSubstitutions = Tuple[Dict[str, str], Dict[str, str]]


def learn_policy(domain_str: str, problem_strs: List[str],
                 plan_strs: List[Plan]) -> LiftedDecisionList:
    """Learn a goal-conditioned policy using Popper."""
    # Parse the PDDL.
    tasks = [Task(domain_str, prob_str) for prob_str in problem_strs]
    domain = tasks[0].domain

    # Since prolog is sensitive to syntax (e.g., no dashes in names), we need
    # to replace some predicate, action, and object names in the tasks. We will
    # invert these substitutions in the learned policies at the end. Note that
    # we don't need to store object name substitutions because objects won't
    # appear in the learned policies.
    domain_substitutions = _get_prolog_domain_substitutions(domain)
    tasks = [_prologify_task(t, domain_substitutions) for t in tasks]

    # Collect all actions seen in the plans; learn one program per action.
    # Actions are recorded with their names and arities.
    action_set: Set[Tuple[str, int, Tuple[str, ...]]] = set()
    for plan in plan_strs:
        for ground_action in plan:
            assert ground_action.startswith("(")
            action_name, remainder = ground_action[1:].split(" ", 1)
            arity = len(remainder.split(" "))
            signature = domain.actions[action_name].signature
            action_types = tuple(t[0].name for _, t in signature)
            action_set.add((action_name, arity, action_types))
    actions = sorted(action_set)
    logging.info(f"Found actions in plans: {actions}")

    # Convert the plans into state-goal-action triplets.
    demo_state_goal_actions = []
    for task, plan in zip(tasks, plan_strs):
        for state, goal, act in utils.plan_to_trajectory(task, plan):
            demo_state_goal_actions.append((state, goal, act))

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
            # just collect seen predicates from the demos themselves.
            bias_str = _create_bias(demo_state_goal_actions, action)
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
            prog = _run_popper(kbpath=temp_dir)
            programs.append(prog)

    # Invert the substitutions so that the policy matches the original domain.
    policy = _popper_programs_to_policy(programs, domain, domain_substitutions)

    return policy


def _run_popper(kbpath: str) -> List:
    """Run popper and return the learned program."""
    logging.debug("Calling popper.")

    # Toggle for debugging.
    # settings = PopperSettings(kbpath=kbpath, debug=True)
    # manager = multiprocessing.Manager()
    # return_dict = manager.dict()
    # _run_popper_process(settings, return_dict)
    # See https://github.com/logic-and-learning-lab/Popper/issues/62
    settings = PopperSettings(kbpath=kbpath)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = Process(target=_run_popper_process, args=(settings, return_dict))
    p.start()
    p.join()
    # End toggle for debugging.

    prog = return_dict['prog']
    if prog is None:
        raise Exception("Popper failed to find a program.")
    return prog


def _run_popper_process(settings: PopperSettings,
                        return_dict: DictProxy) -> None:
    prog, _, _ = learn_solution(settings)
    return_dict['prog'] = prog


def _create_bias(state_action_goals: List[StateGoalAction],
                 action: Tuple[str, int, Tuple[str, ...]]) -> str:
    """Returns the content of a Popper bias file."""
    action_name, action_arity, action_types = action

    # Collect all predicates with their names, arities, and types.
    predicates: Set[Tuple[str, int, Tuple[str, ...]]] = set()
    goal_predicates: Set[Tuple[str, int, Tuple[str, ...]]] = set()
    for state, goal, _ in state_action_goals:
        for atom in state:
            name = atom.name
            arity = len(atom.signature)
            types = tuple(t.name for _, t in atom.signature)
            predicates.add((name, arity, types))
            assert not name.startswith("goal_")
        for atom in goal:
            name = atom.name
            arity = len(atom.signature)
            types = tuple(t[0].name for _, t in atom.signature)
            goal_predicates.add((name, arity, types))

    # Create predicate and goal predicate strings.
    pred_str = "\n".join(f"body_pred({name},{arity+1})."
                         for name, arity, _ in sorted(predicates))
    goal_pred_str = "\n".join(f"body_pred(goal_{name},{arity+1})."
                              for name, arity, _ in sorted(goal_predicates))

    # Add bias for types.
    type_strs: Set[str] = set()
    for name, _, types in predicates | goal_predicates:
        inner_str = ",".join(types)
        line = f"type({name},({inner_str},ex_id))."
        type_strs.add(line)
    inner_str = ",".join(action_types)
    line = f"type({action_name},({inner_str},ex_id))."
    type_strs.add(line)

    type_str = "\n".join(type_strs)

    return f"""% Predicates
{pred_str}

% Goal predicates
{goal_pred_str}

% Action
head_pred({action_name},{action_arity+1}).

% Type constraints
{type_str}

% Example ID can only appear once
:- clause(C), #count{{V : clause_var(C,V),var_type(C,V,ex_id)}} != 1.
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
    # Replace dashes with underscores.
    s = s.replace("-", "_")
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
                     tasks: List[Task], action: Tuple[str, int,
                                                      Tuple[str, ...]]) -> str:
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

    pos_strs: Set[str] = set()
    neg_strs: Set[str] = set()
    for ex_id, (_, _, a) in enumerate(demo_state_goal_actions):
        for other_action in all_possible_actions:
            # Positive example
            if a == other_action:
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


def _popper_programs_to_policy(
        popper_programs: List, domain: PyperplanDomain,
        domain_substitutions: _DomainSubstitutions) -> LiftedDecisionList:
    predicate_subs, operator_subs = domain_substitutions
    inv_pred_subs = {v: k for k, v in predicate_subs.items()}
    inv_op_subs = {v: k for k, v in operator_subs.items()}
    policy_rules = []
    for prog in popper_programs:
        for rule in order_prog(prog):
            act_literal, body = order_rule(rule)
            # Parse action.
            assert act_literal
            action_str = str(act_literal.predicate)
            action_str = inv_op_subs[action_str]
            action_arg_strs = [str(a) for a in act_literal.arguments]
            # By convention, the last argument encodes the example ID, and
            # should be removed.
            example_id_arg = action_arg_strs.pop(-1)
            action = (action_str, action_arg_strs)
            # Parse conditions.
            conditions = []
            for cond in body:
                pred_str = str(cond.predicate)
                if pred_str.startswith("goal_"):
                    pred_str = "goal_" + inv_pred_subs[pred_str[len("goal_"):]]
                else:
                    pred_str = inv_pred_subs[pred_str]
                pred_arg_strs = [
                    str(a) for a in cond.arguments if str(a) != example_id_arg
                ]
                conditions.append((pred_str, pred_arg_strs))
            rule = _create_ldl_rule(conditions, action, domain)
            print(rule)
            policy_rules.append(rule)
    return LiftedDecisionList(policy_rules)


def _create_ldl_rule(conditions: List[Tuple[str, List[str]]],
                     action: Tuple[str, List[str]],
                     domain: PyperplanDomain) -> LDLRule:
    action_name, action_args = action
    name = action_name + "-rule"
    state_preconditions: Set[PyperplanPredicate] = set()
    goal_preconditions: Set[PyperplanPredicate] = set()
    for pred_name, params in conditions:
        if pred_name.startswith("goal_"):
            destination = goal_preconditions
            pred_name = pred_name[len("goal_"):]
        else:
            destination = state_preconditions
        pred = domain.predicates[pred_name]
        orig_signature = pred.signature
        assert len(params) == len(orig_signature)
        sub = {old: new for (old, _), new in zip(orig_signature, params)}
        new_pred = utils.apply_substitution(pred, sub)
        destination.add(new_pred)

    orig_operator = domain.actions[action_name]
    orig_signature = orig_operator.signature
    assert len(action_args) == len(orig_signature)
    sub = {old: new for (old, _), new in zip(orig_signature, action_args)}
    new_operator = utils.apply_substitution(orig_operator, sub)

    # Collect params from the other components.
    new_params_dict = {}
    for cond in state_preconditions | goal_preconditions:
        for param, typ in cond.signature:
            new_params_dict[param] = typ
    for param, typ in new_operator.signature:
        new_params_dict[param] = typ
    new_params = sorted(new_params_dict.items())

    return LDLRule(name, new_params, state_preconditions, goal_preconditions,
                   new_operator)


def _prolog_transform(s: str) -> str:
    return s.replace("-", "_")


def _get_prolog_domain_substitutions(
        domain: PyperplanDomain) -> _DomainSubstitutions:
    predicate_subs: Dict[str, str] = {}
    operator_subs: Dict[str, str] = {}

    for predicate in domain.predicates:
        predicate_subs[predicate] = _prolog_transform(predicate)

    for operator in domain.actions:
        operator_subs[operator] = _prolog_transform(operator)

    assert len(set(predicate_subs.values())) == len(predicate_subs)
    assert len(set(operator_subs.values())) == len(operator_subs)

    return (predicate_subs, operator_subs)


def _prologify_task(task: Task, domain_subs: _DomainSubstitutions) -> Task:
    predicate_subs, operator_subs = domain_subs
    # Apply substitutions to domain.
    domain_str = task.domain_str
    for old, new in predicate_subs.items():
        domain_str = domain_str.replace(old, new)
    for old, new in operator_subs.items():
        domain_str = domain_str.replace(old, new)
    # This might break in some situations, need to be careful.
    domain_str = domain_str.replace("-", "_")
    domain_str = domain_str.replace(" _ ", " - ")

    # Apply substitutions to problem, and handle object names.
    problem_str = task.problem_str
    for old, new in predicate_subs.items():
        problem_str = problem_str.replace(old, new)
    for old, new in operator_subs.items():
        problem_str = problem_str.replace(old, new)
    # This might break in some situations, need to be careful.
    problem_str = problem_str.replace("-", "_")
    problem_str = problem_str.replace(" _ ", " - ")

    return Task(domain_str, problem_str)
