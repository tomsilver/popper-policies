"""Main entry point for experiments."""

import logging
import sys
import time
from typing import List, Tuple

from popper_policies import utils
from popper_policies.envs import create_tasks
from popper_policies.flags import FLAGS, parse_flags
from popper_policies.structs import Plan, Task


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
        demo = (task, plan)
        demos.append(demo)

    del eval_tasks, demos

    script_time = time.time() - script_start
    logging.info(f"\n\nMain script terminated in {script_time:.5f} seconds")


if __name__ == "__main__":  # pragma: no cover
    _main()
