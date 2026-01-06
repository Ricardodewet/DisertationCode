"""Simulated learner environment for adaptive assessment.

This environment models the interaction between a learner and an assessment
system.  The learner has a latent ability and engagement level which
determine their probability of answering correctly and how they react to
question difficulty.  The action space consists of three discrete
difficulty levels (Easy, Medium, Hard).  The state returned to the agent is
a normalised vector capturing the learner’s current ability, engagement,
previous action and previous correctness.

The environment follows a simplified OpenAI Gym interface (`reset` and
`step`) but does not depend on Gym itself.  Episodes run for a fixed number
of interactions.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict

from ..utils.seeding import set_seed


class LearnerEnv:
    """A toy environment modelling a learner’s response to adaptive questions.

    Parameters
    ----------
    max_steps:
        The number of questions (time steps) per episode.
    seed:
        Optional random seed for reproducibility.
    """

    # Difficulty levels mapped to numeric values
    ACTIONS = np.array([0.0, 1.0, 2.0])  # Easy=0, Medium=1, Hard=2

    def __init__(self, max_steps: int = 20, seed: Optional[int] = None) -> None:
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        set_seed(seed)
        self._reset_internal()

    def _reset_internal(self) -> None:
        # Latent ability initialised from truncated normal distribution
        ability = self.rng.normal(loc=1.0, scale=0.3)
        self.ability = float(np.clip(ability, 0.0, 2.0))
        # Engagement starts fully engaged
        self.engagement = 1.0
        # Last action one‑hot vector
        self.last_action_onehot = np.zeros(3, dtype=np.float32)
        # Last correctness indicator
        self.last_correct = 0.0
        # Episode step counter
        self.current_step = 0

    def reset(self) -> np.ndarray:
        """Reset the environment to the beginning of an episode.

        Returns
        -------
        state : ndarray
            Normalised initial state vector.
        """
        self._reset_internal()
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Construct the normalised state vector.

        The state includes scaled ability, engagement, one‑hot encoding of the
        last action, and last correctness.
        """
        ability_norm = self.ability / 2.0  # scale to [0,1]
        state = np.concatenate([
            np.array([ability_norm, self.engagement], dtype=np.float32),
            self.last_action_onehot.astype(np.float32),
            np.array([self.last_correct], dtype=np.float32),
        ])
        return state

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        """Take an action and return the next state, reward, done flag and info.

        Parameters
        ----------
        action_idx:
            Index of the chosen difficulty (0=Easy, 1=Medium, 2=Hard).

        Returns
        -------
        state : ndarray
            Next state vector.
        reward : float
            Reward obtained from this interaction.
        done : bool
            Whether the episode has terminated.
        info : dict
            Additional diagnostic information such as correctness,
            response time and raw ability.
        """
        assert 0 <= action_idx < len(self.ACTIONS), "Invalid action index"
        d = self.ACTIONS[action_idx]
        # Compute probability of answering correctly
        prob_correct = 1.0 / (1.0 + np.exp(-(self.ability - d)))
        # Draw correctness from Bernoulli
        correct = 1.0 if self.rng.random() < prob_correct else 0.0
        # Response time increases with difficulty mismatch
        response_time = 1.0 + 0.5 * abs(d - self.ability)
        # Update ability: increase on correct, decrease slightly on incorrect
        if correct > 0.5:
            self.ability = float(np.clip(self.ability + 0.05, 0.0, 2.0))
        else:
            self.ability = float(np.clip(self.ability - 0.02, 0.0, 2.0))
        # Update engagement: penalise mismatch, reward correct answers
        self.engagement = float(np.clip(
            self.engagement - 0.1 * abs(d - self.ability) + (0.05 if correct > 0.5 else 0.0),
            0.0, 1.0))
        # Compute reward components
        r_c = correct  # correctness reward
        r_f = 1.0 - abs(d - self.ability) / 2.0  # flow/ZPD bonus in [0,1]
        r_t = - response_time / 3.0  # response time penalty
        reward = r_c + r_f + r_t
        # Update last action/correctness for next state
        self.last_action_onehot = np.zeros(3, dtype=np.float32)
        self.last_action_onehot[action_idx] = 1.0
        self.last_correct = correct
        # Increment step and check termination
        self.current_step += 1
        done = self.current_step >= self.max_steps
        next_state = self._get_state()
        info = {
            'correct': float(correct),
            'response_time': float(response_time),
            'ability': float(self.ability),
            'engagement': float(self.engagement),
            'prob_correct': float(prob_correct),
            'difficulty': float(d),
        }
        return next_state, float(reward), done, info
