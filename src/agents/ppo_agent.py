"""Minimal implementation of Proximal Policy Optimisation (PPO).

This agent uses simple linear models for both the policy (action logits)
and value function.  Training is performed with gradient descent on the
clipped PPO objective.  The implementation avoids any external deep
learning frameworks and relies solely on NumPy.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


class PPOAgent:
    """A lightweight PPO agent with linear policy and value function."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.01,
        gamma: float = 0.99,
        lam: float = 0.95,
        eps_clip: float = 0.2,
        value_coef: float = 0.5,
        epochs: int = 3,
        batch_size: int = 64,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.epochs = epochs
        self.batch_size = batch_size
        # Initialise policy weights and biases small
        self.Wp = np.random.randn(action_dim, state_dim) * 0.1
        self.bp = np.zeros(action_dim)
        # Initialise value weights and bias
        self.Wv = np.random.randn(state_dim) * 0.1
        self.bv = 0.0

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    def act(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Sample an action given the current state.

        Returns the action index, log probability and state value.
        """
        logits = self.Wp.dot(state) + self.bp  # shape (action_dim,)
        probs = self._softmax(logits)
        # Sample action
        action = np.random.choice(self.action_dim, p=probs)
        log_prob = np.log(probs[action] + 1e-8)
        value = float(self.Wv.dot(state) + self.bv)
        return action, float(log_prob), value

    def evaluate(self, states: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute log probabilities and values for a batch of states and actions."""
        logits = states @ self.Wp.T + self.bp  # shape (N, action_dim)
        # Compute probabilities
        exp_z = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_z / exp_z.sum(axis=1, keepdims=True)
        # Log probabilities for the chosen actions
        log_probs = np.log(probs[np.arange(len(states)), actions] + 1e-8)
        # State values
        values = states @ self.Wv + self.bv
        return log_probs, values

    def compute_returns_and_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute discounted returns and GAE advantages."""
        n = len(rewards)
        returns = np.zeros(n)
        advantages = np.zeros(n)
        next_value = 0.0
        gae = 0.0
        for t in reversed(range(n)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]
        # Normalise advantages
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        advantages = (advantages - adv_mean) / adv_std
        return returns, advantages

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
    ) -> None:
        """Update the policy and value networks using collected trajectories."""
        num_samples = len(states)
        # Convert to proper shapes
        states = states.astype(np.float32)
        actions = actions.astype(int)
        old_log_probs = old_log_probs.astype(np.float32)
        returns = returns.astype(np.float32)
        advantages = advantages.astype(np.float32)
        # Mini‑batch SGD
        idxs = np.arange(num_samples)
        for _ in range(self.epochs):
            np.random.shuffle(idxs)
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = idxs[start:end]
                s = states[batch_idx]
                a = actions[batch_idx]
                old_lp = old_log_probs[batch_idx]
                ret = returns[batch_idx]
                adv = advantages[batch_idx]
                # Forward pass
                logits = s @ self.Wp.T + self.bp
                exp_z = np.exp(logits - logits.max(axis=1, keepdims=True))
                probs = exp_z / exp_z.sum(axis=1, keepdims=True)
                # New log probs for chosen actions
                new_log_probs = np.log(probs[np.arange(len(s)), a] + 1e-8)
                # Values
                values = s @ self.Wv + self.bv
                # Ratios
                ratios = np.exp(new_log_probs - old_lp)
                # Surrogate objective
                surr1 = ratios * adv
                surr2 = np.clip(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv
                # Determine which elements are clipped
                use_clipped = np.where(surr1 < surr2, 0.0, 1.0)
                # Policy gradient
                # Gradient of log probs w.r.t logits
                grad_logits = probs.copy()
                grad_logits[np.arange(len(s)), a] -= 1.0
                # Multiply by scaling factor: adv * ratios when not clipped, else 0
                scaling = np.where(use_clipped == 0.0, ratios * adv, 0.0)[:, None]
                grad_logit_weighted = scaling * grad_logits
                # Gradients for Wp and bp
                grad_Wp = grad_logit_weighted.T @ s / len(s)
                grad_bp = grad_logit_weighted.mean(axis=0)
                # Value loss gradient: 0.5 * (v - return)^2
                diff = values - ret
                grad_Wv = (diff[:, None] * s).mean(axis=0)
                grad_bv = diff.mean()
                # Update parameters
                self.Wp -= self.lr * grad_Wp
                self.bp -= self.lr * grad_bp
                self.Wv -= self.lr * self.value_coef * grad_Wv
                self.bv -= self.lr * self.value_coef * grad_bv
