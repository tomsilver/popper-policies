"""Main entry point for experiments."""

import logging
import sys
import time
from typing import List, Tuple

from popper_policies import utils
from popper_policies.envs import create_tasks
from popper_policies.flags import FLAGS, parse_flags
from popper_policies.learn import learn_policy
from popper_policies.structs import LiftedDecisionList, Plan, Task


def _main() -> None:
    # Basic setup.
    script_start = time.time()
    str_args = " ".join(sys.argv)
    # Parse command-line flags.
    parse_flags()
    # Set up logging.
    logging.basicConfig(level=FLAGS.loglevel,
                        format="%(message)s",
                        handlers=[logging.StreamHandler()])
    logging.info(f"Running command: python {str_args}")
    logging.info("Full config:")
    logging.info(FLAGS)
    logging.info(f"Git commit hash: {utils.get_git_commit_hash()}")

    # Create training and evaluation tasks.
    logging.info("Generating tasks.")
    train_tasks, eval_tasks = create_tasks(
        env_name=FLAGS.env,
        num_train=FLAGS.num_train_tasks,
        num_eval=FLAGS.num_eval_tasks,
    )

    # Create plans for learning using train tasks.
    logging.info("Creating demos for policy learning.")
    demos: List[Tuple[Task, Plan]] = []
    for task in train_tasks:
        plan, _ = utils.run_planning(task, planner=FLAGS.planner)
        assert plan is not None, "Planning failed"
        logging.debug("DEMO PLAN:")
        for action in plan:
            logging.debug(action)
        demo = (task, plan)
        demos.append(demo)

    # Use demonstrations to learn policy.
    domain_str = demos[0][0].domain_str
    problem_strs = []
    plan_strs = []
    for task, plan in demos:
        assert task.domain_str == domain_str
        problem_strs.append(task.problem_str)
        plan_strs.append(plan)
    policy = learn_policy(domain_str, problem_strs, plan_strs,
                          FLAGS.popper_max_body, FLAGS.popper_max_vars)
    logging.info(f"Learned policy:\n{policy}")

    # Evaluate the learned policy.
    num_solved = 0
    for task in eval_tasks:
        solved = _evaluate_policy(task, policy)
        if solved:
            num_solved += 1
    num_tasks = len(eval_tasks)
    logging.info(f"Policy solved {num_solved} out of {num_tasks} eval tasks.")

    script_time = time.time() - script_start
    logging.info(f"\n\nMain script terminated in {script_time:.5f} seconds")


def _evaluate_policy(task: Task, policy: LiftedDecisionList) -> bool:
    goal = set(utils.get_goal_strs(task))
    objects = set(task.problem.objects.items()) | set(
        task.domain.constants.items())
    for _ in range(FLAGS.horizon):
        state = set(utils.get_init_strs(task))
        if goal.issubset(state):
            logging.debug("Goal reached!")
            return True
        act = utils.query_ldl(policy, state, objects, goal)
        logging.debug(f"Taking action: {act} from state: {state}")
        if act is None:
            return False
        task = utils.advance_task(task, act)
    return False


if __name__ == "__main__":  # pragma: no cover
    _main()
