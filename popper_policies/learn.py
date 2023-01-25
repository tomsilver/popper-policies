"""Learn policies for PDDL domains using Popper (ILP system)."""

import itertools
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
    PyperplanDomain, PyperplanPredicate, PyperplanType, StateGoalAction, \
    Task

# Predicate substitutions, action substitutions.
_DomainSubstitutions = Tuple[Dict[str, str], Dict[str, str]]


def learn_policy(domain_str: str,
                 problem_strs: List[str],
                 plan_strs: List[Plan],
                 popper_max_body: int = 10,
                 popper_max_vars: int = 6,
                 planner_name: str = "pyperplan") -> LiftedDecisionList:
    """Learn a goal-conditioned policy using Popper."""
    # Parse the PDDL.
    tasks = [Task(domain_str, prob_str) for prob_str in problem_strs]
    original_domain = tasks[0].domain

    # Since prolog is sensitive to syntax (e.g., no dashes in names), we need
    # to replace some predicate, action, and object names in the tasks. We will
    # invert these substitutions in the learned policies at the end. Note that
    # we don't need to store object name substitutions because objects won't
    # appear in the learned policies.
    domain_substitutions = _get_prolog_domain_substitutions(original_domain)
    tasks = [_prologify_task(t, domain_substitutions) for t in tasks]
    domain = tasks[0].domain
    domain_str = tasks[0].domain_str

    # Collect all actions seen in the plans; learn one program per action.
    # Actions are recorded with their names and arities.
    action_set: Set[Tuple[str, int, Tuple[str, ...]]] = set()
    for plan in plan_strs:
        for ground_action in plan:
            assert ground_action.startswith("(")
            action_name, remainder = ground_action[1:].split(" ", 1)
            arity = len(remainder.split(" "))
            signature = original_domain.actions[action_name].signature
            action_types = tuple(t[0].name for _, t in signature)
            action_set.add((action_name, arity, action_types))
    actions = sorted(action_set)
    logging.info(f"Found actions in plans: {actions}")

    # Generate all possible ground actions for the sake of negative examples.
    # We may want to subsample in the future because this could get very big.
    all_ground_actions: Set[str] = set()
    for task in tasks:
        for op in task.pyperplan_task.operators:
            all_ground_actions.add(op.name)

    # Convert the plans into state-goal-object-action tuples.
    # Add in negated predicates.
    demo_state_goal_actions = []
    for task, plan in zip(tasks, plan_strs):
        for state, goal, objs, act in utils.plan_to_trajectory(task, plan):
            # Add negated facts to state.
            state_facts = {utils.pred_to_str(p) for p in state}
            for fact in task.pyperplan_task.facts:
                if fact not in state_facts:
                    negated_pred = _create_negated_predicate(
                        fact, task.domain.predicates)
                    state.add(negated_pred)
            demo_state_goal_actions.append((state, goal, objs, act))

    # Won't need the tasks anymore.
    del tasks

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
            bias_str = _create_bias(demo_state_goal_actions, action, domain,
                                    popper_max_body, popper_max_vars)
            logging.debug(f"Created bias string:\n{bias_str}")
            bias_file = temp_dir_path / "bias.pl"
            with open(bias_file, "w", encoding="utf-8") as f:
                f.write(bias_str)

            # Create the background knowledge (bk) file.
            bk_file = temp_dir_path / "bk.pl"
            with open(bk_file, "w", encoding="utf-8") as f:
                f.write(bk_str)

            # Create the examples (exs) file.
            examples_str = _create_examples(demo_state_goal_actions,
                                            all_ground_actions, action,
                                            domain.name, domain_str,
                                            planner_name)
            logging.debug(f"Created examples string:\n{examples_str}")
            examples_file = temp_dir_path / "exs.pl"
            with open(examples_file, "w", encoding="utf-8") as f:
                f.write(examples_str)

            # Call popper.
            prog = _run_popper(kbpath=temp_dir)
            programs.append(prog)

    # Invert the substitutions so that the policy matches the original domain.
    policy = _popper_programs_to_policy(programs, original_domain,
                                        domain_substitutions)

    return policy


def _run_popper(kbpath: str) -> List:
    """Run popper and return the learned program."""
    logging.debug("Calling popper.")

    # Toggle for debugging.
    # settings = PopperSettings(kbpath=kbpath, debug=True, explain=True)
    # return_dict = {}
    # _run_popper_process(settings, return_dict)
    # See https://github.com/logic-and-learning-lab/Popper/issues/62
    settings = PopperSettings(kbpath=kbpath, explain=True)
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
                 action: Tuple[str, int, Tuple[str, ...]],
                 domain: PyperplanDomain, max_body: int, max_vars: int) -> str:
    """Returns the content of a Popper bias file."""
    action_name, action_arity, action_types = action

    # Collect all predicates with their names, arities, and types.
    predicates: Set[Tuple[str, int, Tuple[str, ...]]] = set()
    goal_predicates: Set[Tuple[str, int, Tuple[str, ...]]] = set()
    for state, goal, _, _ in state_action_goals:
        for atom in state:
            name = atom.name
            arity = len(atom.signature)
            types = utils.pred_to_type_names(atom)
            predicates.add((name, arity, types))
            assert not name.startswith("goal_")
        for atom in goal:
            name = atom.name
            arity = len(atom.signature)
            types = utils.pred_to_type_names(atom)
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

    # Add bias for operator preconditions.
    preconditions_strs: Set[str] = set()
    counter = itertools.count()
    var_to_idx = {}
    action_signature = domain.actions[action_name].signature
    preconditions = domain.actions[action_name].precondition
    for v, _ in action_signature:
        var_to_idx[v] = next(counter)
    var_to_idx["ex_id"] = next(counter)
    for precond in preconditions:
        name = precond.name
        arity = len(precond.signature) + 1  # plus 1 for experiment id
        var_idxs = [var_to_idx[v] for v, _ in precond.signature]
        var_idxs += [var_to_idx["ex_id"]]
        var_str = ",".join(map(str, var_idxs))
        precond_str = f":- not body_literal(0,{name},{arity},({var_str}))."
        preconditions_strs.add(precond_str)

    preconditions_str = "\n".join(preconditions_strs)

    return f"""% Set max bounds
max_body({max_body}).
max_vars({max_vars}).

% Predicates
{pred_str}

% Goal predicates
{goal_pred_str}

% Action
head_pred({action_name},{action_arity+1}).

% Type constraints
{type_str}

% Example ID can only appear once
:- clause(C), #count{{V : clause_var(C,V),var_type(C,V,ex_id)}} != 1.

% Action preconditions (and suppress ASP warning)
#defined body_literal/4.
{preconditions_str}
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

    for example_id, (state, goal, _, _) in enumerate(demo_state_goal_actions):
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
                     all_ground_actions: Set[str],
                     action: Tuple[str, int, Tuple[str,
                                                   ...]], domain_name: str,
                     domain_str: str, planner_name: str) -> str:
    """Returns the content of a Popper examples file.

    Detects and filters false negatives by planning for each possible
    successor and checking to see if the result is worse than the
    demonstration.
    """
    pos_strs: Set[str] = set()
    neg_strs: Set[str] = set()

    all_matching_actions = {
        a
        for a in all_ground_actions if a.startswith(f"({action[0]}")
    }
    for ex_id, (s, g, o, a) in enumerate(demo_state_goal_actions):
        # Get the cost-to-go for this state.
        demo_ctg = _get_cost_to_go(s, g, o, a, domain_name, domain_str,
                                   planner_name)
        assert demo_ctg < float("inf")
        for other_action in all_matching_actions:
            # Positive example
            if a == other_action:
                wrapper = "pos"
                destination = pos_strs
            # Negative example
            else:
                # Detect false negatives.
                ctg = _get_cost_to_go(s, g, o, other_action, domain_name,
                                      domain_str, planner_name)
                if ctg <= demo_ctg:
                    continue
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
    pos_state_preconditions: Set[PyperplanPredicate] = set()
    neg_state_preconditions: Set[PyperplanPredicate] = set()
    goal_preconditions: Set[PyperplanPredicate] = set()
    for pred_name, params in conditions:
        if pred_name.startswith("goal_"):
            destination = goal_preconditions
            pred_name = pred_name[len("goal_"):]
        elif pred_name.startswith("negated_"):
            destination = neg_state_preconditions
            pred_name = pred_name[len("negated_"):]
        else:
            destination = pos_state_preconditions
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
    for cond in pos_state_preconditions | neg_state_preconditions | \
                goal_preconditions:
        for param, typ in cond.signature:
            new_params_dict[param] = typ
    for param, typ in new_operator.signature:
        new_params_dict[param] = typ
    new_params = sorted(new_params_dict.items())

    return LDLRule(name, new_params, pos_state_preconditions,
                   neg_state_preconditions, goal_preconditions, new_operator)


def _prolog_transform(s: str) -> str:
    return s.replace("-", "_")


def _get_prolog_domain_substitutions(
        domain: PyperplanDomain) -> _DomainSubstitutions:
    predicate_subs: Dict[str, str] = {}
    operator_subs: Dict[str, str] = {}

    for predicate in domain.predicates:
        predicate_subs[predicate] = _prolog_transform(predicate)
        # Add negated versions too.
        assert "negated_" not in predicate
        predicate_subs["negated_" + predicate] = _prolog_transform("negated_" +
                                                                   predicate)

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


def _create_negated_predicate(fact: str, predicates: Dict[str,
                                                          PyperplanPredicate]):
    pred = utils.str_to_pred(fact, predicates)
    pred.name = "negated_" + pred.name
    return pred


def _get_cost_to_go(state: Set[PyperplanPredicate],
                    goal: Set[PyperplanPredicate],
                    objects: Dict[str, PyperplanType], action: str,
                    domain_name: str, domain_str: str,
                    planner_name: str) -> float:
    # Create a task.
    objects_str = "\n  ".join([f"{o} - {t}" for o, t in objects.items()])
    init_str = "\n  ".join([
        utils.pred_to_str(p) for p in state
        if not p.name.startswith("negated_")
    ])
    goal_str = "\n  ".join([utils.pred_to_str(p) for p in goal])
    problem_str = f"""(define (problem synthetic-problem)
   (:domain {domain_name})
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
    task = Task(domain_str, problem_str)

    # If the action is invalid, treat as infinite cost to go.
    if not utils.action_is_valid_for_task(task, action):
        return float("inf")

    # Advance via the action.
    task = utils.advance_task(task, action)

    # Plan to the goal.
    plan, _ = utils.run_planning(task, planner=planner_name)
    assert plan is not None, "Planning failed"

    # Measure the cost.
    return len(plan)
