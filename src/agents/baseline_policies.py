"""Baseline policies for adaptive question selection.

Each policy is a callable that, given the current learner environment, returns
an action index (0=Easy, 1=Medium, 2=Hard).  Baselines do not learn; they
implement fixed strategies inspired by the literature.
"""

from __future__ import annotations

import random
from typing import Callable

from ..envs.learner_env import LearnerEnv


def random_policy(env: LearnerEnv) -> int:
    """Select a difficulty uniformly at random."""
    return random.randint(0, 2)


def static_medium_policy(env: LearnerEnv) -> int:
    """Always choose the medium difficulty (index 1)."""
    return 1


def heuristic_zpd_policy(env: LearnerEnv) -> int:
    """Select difficulty based on current ability.

    If ability < 0.7 → Easy, ability ∈ [0.7, 1.3] → Medium, else Hard.
    """
    ability = env.ability
    if ability < 0.7:
        return 0
    elif ability < 1.3:
        return 1
    else:
        return 2


class StaircasePolicy:
    """A staircase policy that adapts difficulty based on correctness.

    The policy starts at Medium.  After each correct answer it increases
    difficulty by one level (up to Hard); after each incorrect answer it
    decreases difficulty by one level (down to Easy).
    """

    def __init__(self) -> None:
        self.current_action: int = 1  # start at Medium

    def __call__(self, env: LearnerEnv) -> int:
        return self.current_action

    def update(self, correct: float) -> None:
        if correct > 0.5:
            self.current_action = min(self.current_action + 1, 2)
        else:
            self.current_action = max(self.current_action - 1, 0)
