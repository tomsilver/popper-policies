"""Learn policies for PDDL domains using Popper (ILP system)."""

from typing import List

from popper_policies.structs import Plan


def learn_policy(domain_str: str, problem_strs: List[str],
                 plan_strs: List[Plan]) -> str:
    """Learn a goal-conditioned policy using Popper."""
    del domain_str, problem_strs, plan_strs
    return "TODO"
