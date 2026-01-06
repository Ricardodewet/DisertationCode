"""Tests for baseline policies."""

from __future__ import annotations

from ..src.envs.learner_env import LearnerEnv
from ..src.agents.baseline_policies import (
    random_policy,
    static_medium_policy,
    heuristic_zpd_policy,
    StaircasePolicy,
)


def test_baseline_actions():
    env = LearnerEnv(max_steps=1, seed=1)
    env.reset()
    # Random policy returns valid index
    a = random_policy(env)
    assert 0 <= a <= 2
    # Static always medium
    assert static_medium_policy(env) == 1
    # Heuristic selects easy for low ability
    env.ability = 0.5
    assert heuristic_zpd_policy(env) == 0
    env.ability = 1.0
    assert heuristic_zpd_policy(env) == 1
    env.ability = 1.6
    assert heuristic_zpd_policy(env) == 2


def test_staircase_policy():
    stair = StaircasePolicy()
    # Start at medium
    assert stair(None) == 1
    # Correct answer increases difficulty
    stair.update(1.0)
    assert stair(None) == 2
    # Incorrect answer decreases difficulty
    stair.update(0.0)
    assert stair(None) == 1
