"""Spectral augmentation utilities for hyperspectral cubes."""

# src/data/augmentations.py
import numpy as np

def random_band_dropout(cube: np.ndarray, max_frac: float = 0.1):
    D = cube.shape[-1]
    k = np.random.randint(1, int(D * max_frac) + 1)
    bands = np.random.choice(D, k, replace=False)
    cube = cube.copy()
    cube[..., bands] = 0.0
    return cube

def spectral_jitter(cube: np.ndarray, sigma: float = 0.01):
    noise = np.random.normal(scale=sigma, size=cube.shape)
    return cube + noise

def spectral_shuffle(cube: np.ndarray, max_shuffle: int = 3):
    D = cube.shape[-1]
    idx = np.arange(D)
    for _ in range(np.random.randint(1, max_shuffle + 1)):
        i = np.random.randint(0, D - 1)
        idx[i], idx[i + 1] = idx[i + 1], idx[i]
    return cube[..., idx]