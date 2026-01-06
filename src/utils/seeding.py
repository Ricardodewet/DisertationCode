"""Utilities for reproducible random number generation.

The simulation relies on Python's built‑in `random` module and NumPy's
random generator.  This helper ensures that both are seeded consistently.
"""

from __future__ import annotations

import random
import numpy as np
from typing import Optional


def set_seed(seed: Optional[int]) -> None:
    """Seed both the built‑in random module and NumPy.

    If ``seed`` is ``None`` the generators are left in their current state.

    Parameters
    ----------
    seed:
        Integer seed to initialise the pseudo‑random number generators.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
