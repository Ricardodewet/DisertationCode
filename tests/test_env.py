"""Unit tests for the learner environment."""

from __future__ import annotations

import numpy as np

from ..src.envs.learner_env import LearnerEnv


def test_environment_step_and_reset():
    env = LearnerEnv(max_steps=5, seed=123)
    state = env.reset()
    # initial state values within [0,1]
    assert 0.0 <= state[0] <= 1.0  # ability
    assert 0.0 <= state[1] <= 1.0  # engagement
    # take a step with action 1 (medium)
    next_state, reward, done, info = env.step(1)
    # check ranges
    assert isinstance(reward, float)
    assert 0.0 <= next_state[0] <= 1.0
    assert 0.0 <= info['engagement'] <= 1.0
    assert info['difficulty'] in [0.0, 1.0, 2.0]
    assert 0.0 <= info['prob_correct'] <= 1.0
    assert info['response_time'] >= 1.0
    # ensure done becomes True after max_steps
    steps = 1
    while not done:
        _, _, done, _ = env.step(1)
        steps += 1
    assert steps == env.max_steps


def test_seeding_reproducibility():
    env1 = LearnerEnv(max_steps=3, seed=42)
    env2 = LearnerEnv(max_steps=3, seed=42)
    s1 = env1.reset()
    s2 = env2.reset()
    assert np.allclose(s1, s2)
    for action in [0, 1, 2]:
        n1, r1, d1, _ = env1.step(action)
        n2, r2, d2, _ = env2.step(action)
        assert np.allclose(n1, n2)
        assert r1 == r2
        assert d1 == d2
