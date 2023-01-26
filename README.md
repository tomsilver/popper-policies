# Goal-conditioned policy learning in PDDL domains with Popper (ILP)

**Objective:** Given a PDDL domain, example problems, and example plans (one per problem), learn a goal-conditioned policy that generalizes to other problems in the domain.

**Warning:** This repository is extremely experimental. It probably won't work for your use case.

## Installation

1. Follow the instructions to install [Popper](https://github.com/logic-and-learning-lab/Popper).
> The code in this repository was developed on Popper commit `bcf8b8de8b8de97f09ca125eb6e1cce11bc4f310`. If you have trouble running Popper, try to install from that commit.

2. Clone this repository.
3. In the cloned repository: `pip install -e .`
4. Optional but recommended for speed: install Fast Downward as described below.
    1. `git clone https://github.com/aibasel/downward.git`
    2) `cd downward && ./build.py`
    3) `export FD_EXEC_PATH="<your absolute path here>/downward"`

## Example

```python
from popper_policies.learn import learn_policy

domain_str = """(define (domain toy-delivery)
    (:requirements :strips)
    (:predicates 
        (at-robby ?loc)
        (satisfied ?loc)
    )
    
    (:action move
        :parameters (?from ?to)
        :precondition (and
            (at-robby ?from) 
        )
        :effect (and
            (not (at-robby ?from))
            (at-robby ?to)
        )
    )
    
    (:action deliver
        :parameters (?loc)
        :precondition (and
            (at-robby ?loc)
        )
        :effect (and
            (satisfied ?loc)
        )
    )
    
)
"""

problem1 = """(define (problem toy-delivery) (:domain toy-delivery)
  (:objects
    loc0 loc1 loc2
  )
  (:init 
	(at-robby loc0)
  )
  (:goal (and (satisfied loc1)))
)
"""
plan1 = [
    "(move loc0 loc1)",
    "(deliver loc1)",
]

problem2 = """(define (problem toy-delivery) (:domain toy-delivery)
  (:objects
    loc0 loc1 loc2
  )
  (:init 
	(at-robby loc0)
  )
  (:goal (and (satisfied loc2)))
)
"""
plan2 = [
    "(move loc0 loc2)",
    "(deliver loc2)",
]

policy = learn_policy(domain_str, [problem1, problem2], [plan1, plan2])

print(policy)
```

Output:

```lisp
(define (policy)
  (:rule deliver-rule
    :parameters (?A - object)
    :preconditions (at-robby ?A)
    :goals (satisfied ?A)
    :action (deliver ?A)
  )
  (:rule move-rule
    :parameters (?A - object ?B - object)
    :preconditions (and (at-robby ?A) (not (at-robby ?B)))
    :goals (satisfied ?B)
    :action (move ?A ?B)
  )
)
```

See `popper_policies/main.py` for automatically generating demonstrations and evaluating the learned policy. For example:

```
python popper_policies/main.py --env custom-toy --seed 0 --num_train_tasks 2 --num_eval_tasks 1
```