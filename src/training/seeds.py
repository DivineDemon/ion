"""Seed setting for reproducibility. Call at the start of every run."""

import random
from typing import Any

import numpy as np
import torch


def set_seeds(config: dict[str, Any]) -> None:
    """
    Set global random seeds from config for reproducibility.
    Uses config['seed']; if missing, defaults to 42.
    """
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = True
