"""Tests for the PPO agent implementation."""

from __future__ import annotations

import numpy as np
from ..src.agents.ppo_agent import PPOAgent


def test_act_and_update():
    state_dim = 7
    action_dim = 3
    agent = PPOAgent(state_dim, action_dim, learning_rate=0.01)
    # Generate dummy data
    num_samples = 10
    states = np.random.rand(num_samples, state_dim).astype(np.float32)
    actions = np.random.randint(0, action_dim, size=num_samples)
    old_log_probs = np.zeros(num_samples)
    values = np.zeros(num_samples)
    rewards = np.random.randn(num_samples).astype(np.float32)
    dones = np.zeros(num_samples)
    # Compute returns and advantages
    returns, advantages = agent.compute_returns_and_advantages(rewards, values, dones)
    # Before update, act should return a valid action
    action, logp, value = agent.act(states[0])
    assert 0 <= action < action_dim
    assert isinstance(logp, float)
    assert isinstance(value, float)
    # Update without error
    agent.update(states, actions, old_log_probs, returns, advantages)
    # After update, parameters should change slightly
    # We cannot strictly test for improvement but ensure no NaNs
    assert not np.isnan(agent.Wp).any()
    assert not np.isnan(agent.Wv).any()
