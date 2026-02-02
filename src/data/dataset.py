# src/data/dataset.py

"""
PyTorch Dataset for hyperspectral-image classification (patch-based).
This module provides a `HyperspectralDataset` class that can be used to load
hyperspectral image cubes and their corresponding ground truth labels.
It supports patch extraction, data augmentation, and normalization.
It is designed to work with datasets like Houston13, Pavia_University, and Salinas.
It is compatible with PyTorch's `Dataset` and `DataLoader` classes.
This code is part of the GAST project, which implements GAST for hyperspectral image classification.
"""

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset
from pathlib import Path
import os


from src.data.augmentations import (
    random_band_dropout, spectral_jitter, spectral_shuffle
)
from src.data.data_loader import load_dataset_from_path, load_gt_from_path


class HyperspectralDataset(Dataset):
    def __init__(
        self,
        cube_path: str,
        gt_path: str,
        indices: np.ndarray,
        patch_size: int = 11,
        mode: str = "train",
        mean: np.ndarray | None = None,
        std:  np.ndarray | None = None,
        augment: bool = False,
        stride: int = 1,
        cfg=None,
        preloaded_cube: np.ndarray | None = None, # <-- Add this parameter
    ):
        """
        PyTorch Dataset for hyperspectral-image classification.

        Args:
            cube_path (str): Path to the hyperspectral data cube (.mat file).
            gt_path (str): Path to the ground truth labels (.mat file).
            indices (np.ndarray): Pre-computed indices (N, 2) for the data split. This is a required argument.
            patch_size (int): The height and width of the patches to extract.
            mode (str): One of "train", "val", or "test".
            mean (np.ndarray, optional): Pre-computed mean for normalization. Required for val/test modes.
            std (np.ndarray, optional): Pre-computed standard deviation for normalization. Required for val/test modes.
            augment (bool): If True, apply data augmentation during training.
            stride (int): The stride for patch extraction (not fully implemented here).
            cfg (object, optional): A configuration object (e.g., from argparse).
                This object is used to control specific augmentations. It is expected
                to have the following boolean attributes:
                - `no_band_dropout` (bool): If True, disables random band dropout.
                - `no_spec_jitter` (bool): If True, disables spectral jitter.
                - `no_spec_shuffle` (bool): If True, disables spectral shuffling.
            preloaded_cube (np.ndarray, optional): Pre-loaded cube data. If provided, used instead of loading from cube_path.
        """
        # Convert string paths to Path objects
        cube_path = Path(cube_path)
        gt_path = Path(gt_path)

        if not os.path.exists(cube_path):
            raise FileNotFoundError(f"Cube file not found: {cube_path}")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

        self.mode = mode
        self.patch_size = patch_size
        self.augment = augment and mode == "train"
        # Parse augmentation settings from the config object
        if self.augment and cfg is not None:
            self.use_band_dropout = not getattr(cfg, "no_band_dropout", False)
            self.use_spec_jitter = not getattr(cfg, "no_spec_jitter", False)
            self.use_spec_shuffle = not getattr(cfg, "no_spec_shuffle", False)
        else:
            self.use_band_dropout = False
            self.use_spec_jitter = False
            self.use_spec_shuffle = False

        # --- MODIFICATION START ---
        # Use the provided data if available, otherwise load from files
        if preloaded_cube is not None:
            self.cube = preloaded_cube
            self.gt, _ = load_gt_from_path(gt_path) # Still need to load GT
        else:
            self.cube, self.gt = load_dataset_from_path(cube_path, gt_path)
        # --- MODIFICATION END ---

        self.H, self.W, self.D = self.cube.shape

        # ── Use the provided pixel indices ────────────────────────────────────
        self.indices = np.asarray(indices)
        self.labels = self.gt[self.indices[:, 0], self.indices[:, 1]] - 1  # make labels zero-indexed
        
        # ── stats ────────────────────────────────────────────────────────────
        if mean is not None and std is not None:
            self.mean, self.std = mean, std
        elif mode == "train":
            self.mean, self.std = self._compute_mean_std()
        else:
            raise ValueError("mean/std must be given for val & test sets")

        # pad cube for border patches
        pad = patch_size // 2
        self.pad = pad
        self.padded_cube = np.pad(
            self.cube, ((pad, pad), (pad, pad), (0, 0)), mode="reflect"
        )

    # --------------------------------------------------------------------- #
    #  helper functions
    # --------------------------------------------------------------------- #
    @staticmethod
    def _read_mat(path: str) -> np.ndarray:
        key, = (k for k in scipy.io.loadmat(path).keys() if not k.startswith("__"))
        return scipy.io.loadmat(path)[key]

    def _compute_mean_std(self):
        # stream over pixels to keep memory modest
        n, mean, M2 = 0, np.zeros(self.D), np.zeros(self.D)
        for r, c in self.indices:
            v = self.cube[r, c]                 # (D,)
            n += 1
            delta = v - mean
            mean += delta / n
            M2   += delta * (v - mean)
        var = M2 / max(n - 1, 1)
        return mean, np.sqrt(var) + 1e-8

    def _get_patch(self, r: int, c: int):
        p = self.pad
        patch = self.padded_cube[r : r + 2*p + 1, c : c + 2*p + 1]
        return patch  # (k, k, D)
        
    def _augment(
        self,
        patch: np.ndarray,
        use_band_dropout: bool,
        use_spec_jitter: bool,
        use_spec_shuffle: bool
    ):
        patch = patch.astype(np.float32)
        # Standard spatial augmentations
        if np.random.rand() < 0.5: patch = np.fliplr(patch)
        if np.random.rand() < 0.5: patch = np.flipud(patch)
        patch = np.rot90(patch, k=np.random.randint(4))

        # Spectral augmentations controlled by flags
        if use_band_dropout and np.random.rand() < 0.3:
            patch = random_band_dropout(patch, max_frac=0.1)
        if use_spec_jitter and np.random.rand() < 0.5:
            patch = spectral_jitter(patch, sigma=0.01)
        if use_spec_shuffle and np.random.rand() < 0.3:
            patch = spectral_shuffle(patch, max_shuffle=3)
            
        # Brightness adjustment
        if np.random.rand() < 0.3:
            patch *= 0.9 + 0.2 * np.random.rand()
            
        return patch

    def __len__(self):       return len(self.indices)

    def __getitem__(self, i):
        r, c = self.indices[i]
        patch = self._get_patch(r, c)
        
        if self.augment:
            patch = self._augment(
                patch,
                use_band_dropout=self.use_band_dropout,
                use_spec_jitter=self.use_spec_jitter,
                use_spec_shuffle=self.use_spec_shuffle
            )

        patch = (patch - self.mean) / self.std         # normalise
        patch = np.transpose(patch, (2, 0, 1)).astype(np.float32)

        return {
            "patch": torch.from_numpy(patch),
            "label": torch.tensor(self.labels[i], dtype=torch.long),
            "coord": torch.tensor(self.indices[i], dtype=torch.long),  
        }