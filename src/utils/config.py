"""Configuration loading utilities.

The experiment settings are defined in YAML files.  This module provides
a simple interface for reading those files into Python dictionaries.
"""

from __future__ import annotations

import os
import yaml
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path:
        Path to a YAML file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file {path} does not exist")
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg
