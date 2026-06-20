"""Unified seeding for reproducibility: random, numpy, PYTHONHASHSEED."""

import os
import random

import numpy as np

SEED = 42


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
