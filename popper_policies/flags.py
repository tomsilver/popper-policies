"""Command line flags."""

import argparse
import logging

FLAGS = argparse.Namespace()  # set by parse_flags() below


def parse_flags() -> None:
    """Parse the command line flags and update global FLAGS."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--num_train_tasks", default=5, type=int)
    parser.add_argument("--num_eval_tasks", default=10, type=int)
    parser.add_argument("--horizon", default=100, type=int)
    parser.add_argument("--planner", default="pyperplan")
    parser.add_argument("--popper_max_body", default=10, type=int)
    parser.add_argument("--popper_max_vars", default=6, type=int)
    parser.add_argument('--debug',
                        action="store_const",
                        dest="loglevel",
                        const=logging.DEBUG,
                        default=logging.INFO)
    args = parser.parse_args()
    FLAGS.__dict__.update(args.__dict__)
