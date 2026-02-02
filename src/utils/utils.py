"""General utilities for seeding, PCA, losses, splits, and logging."""

# src/utils/utils.py

import os
import logging
from typing import Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA

# -----------------------------------------------------------------------------#
#  Logging
# -----------------------------------------------------------------------------#
log = logging.getLogger(__name__)

# If the root logger has no handler yet, configure a minimal one so that
#  "python -m src.training.train ..." prints something useful.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)s]  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# -----------------------------------------------------------------------------#
#  Reproducibility helper
# -----------------------------------------------------------------------------#
def set_seed(seed: int = 242) -> None:
    """
    Seed Python `random`, NumPy and PyTorch (both CPU & GPU) RNGs.

    Call this once at the top of your script **before** any RNG is used.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log.info("Global RNG seed set to %d", seed)


# -----------------------------------------------------------------------------#
#  Checkpoint I/O
# -----------------------------------------------------------------------------#
def save_checkpoint(checkpoint: dict, output_dir: str, filename: str) -> None:
    """
    Save *checkpoint* (dict) to *output_dir/filename*.

    Example
    -------
    >>> save_checkpoint(
    ...     {"model_state_dict": model.state_dict(), "epoch": 12},
    ...     "./models", "gast_best.pth")
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    try:
        torch.save(checkpoint, path)
        log.info("Checkpoint saved → %s", path)
    except Exception as e:  # pragma: no cover
        log.error("Failed to save checkpoint to %s: %s", path, e)


def load_checkpoint(filepath: str, device: torch.device | None = None) -> dict | None:
    """
    Load checkpoint dict from *filepath*; returns **None** if loading fails.
    """
    if not os.path.isfile(filepath):
        log.error("Checkpoint not found: %s", filepath)
        return None

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(filepath, map_location=device)
        log.info("Loaded checkpoint ← %s", filepath)
        return ckpt
    except Exception as e:  # pragma: no cover
        log.error("Failed to load checkpoint %s: %s", filepath, e)
        return None


# -----------------------------------------------------------------------------#
#  Quick evaluation metric
# -----------------------------------------------------------------------------#
@torch.no_grad()
def OA(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device | str = "cpu",
    return_cm: bool = False,
) -> float | Tuple[float, np.ndarray]:
    """
    Compute Overall Accuracy on *loader*, ignoring labels == -1.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    device : torch.device or str
    return_cm : bool
        If *True*, also return the confusion matrix (NumPy array).

    Returns
    -------
    oa : float
        Overall accuracy.
    cm : np.ndarray    [optional]
        Confusion matrix of shape (n_classes, n_classes).
    """
    model.eval()
    device = torch.device(device)
    model.to(device)

    total = correct = 0
    cm = None

    for batch in loader:
        x = batch["patch"].to(device)
        y = batch["label"].to(device)

        logits = model(x)
        preds = logits.argmax(1)

        mask = (y != -1)
        correct += (preds[mask] == y[mask]).sum().item()
        total += int(mask.sum().item())

        if return_cm:
            if cm is None:
                n_cls = logits.size(1)
                cm = torch.zeros(n_cls, n_cls, dtype=torch.int64, device=device)
            for t, p in zip(y[mask].view(-1), preds[mask].view(-1)):
                cm[t, p] += 1

    oa = correct / total if total else 0.0
    if return_cm:
        return oa, cm.cpu().numpy()
    return oa

# -----------------------------------------------------------------------------#
#  Stratified split with minimum per-class samples
# -----------------------------------------------------------------------------#
from sklearn.utils import check_random_state
import numpy as np

def stratified_min_samples_split(
    coords: np.ndarray,
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    train_min_frac: float = 0.05,  # minimum fraction for training
    val_min_frac: float = 0.0,     # minimum fraction for validation
    train_floor: int = 5,          # absolute minimum for training
    val_floor: int = 0,            # absolute minimum for validation
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified split with separate minimum sample guarantees for train/val.
    
    Parameters
    ----------
    coords : array of shape (N, 2)
        Coordinates of samples
    labels : array of shape (N,)
        Class labels
    train_ratio : float
        Desired ratio of samples for training
    val_ratio : float
        Desired ratio of samples for validation
    train_min_frac : float
        Minimum fraction of samples per class for training
    val_min_frac : float
        Minimum fraction of samples per class for validation
    train_floor : int
        Absolute minimum samples per class for training
    val_floor : int
        Absolute minimum samples per class for validation
    seed : int, optional
        Random seed for reproducibility
    """
    rng = check_random_state(seed)
    train_list, val_list, test_list = [], [], []

    for cls in np.unique(labels):
        if cls < 0:
            continue  # skip background/ignore

        cls_mask = (labels == cls)
        cls_coords = coords[cls_mask]
        n = len(cls_coords)
        if n == 0:
            continue

        # compute separate floors for train and val
        train_floor_frac = int(np.floor(n * train_min_frac))
        val_floor_frac = int(np.floor(n * val_min_frac))
        
        floor_train = max(train_floor_frac, train_floor)
        floor_val = max(val_floor_frac, val_floor)

        # desired counts before clamping
        desired_train = int(np.floor(n * train_ratio))
        desired_val = int(np.floor(n * val_ratio))

        # apply separate floors
        n_train = max(desired_train, floor_train)
        n_val = max(desired_val, floor_val)

        # clamp so we don't exceed available samples
        n_train = min(n_train, n - 2)
        n_val = min(n_val, n - n_train - 1)

        perm = rng.permutation(n)
        train_i = perm[:n_train]
        val_i = perm[n_train:n_train + n_val]
        test_i = perm[n_train + n_val:]

        train_list.append(cls_coords[train_i])
        val_list.append(cls_coords[val_i])
        test_list.append(cls_coords[test_i])

    return (
        np.vstack(train_list),
        np.vstack(val_list),
        np.vstack(test_list),
    )


# -----------------------------------------------------------------------------#
#  Focal Loss for class imbalance
# -----------------------------------------------------------------------------#
import torch
import torch.nn.functional as F
from torch import nn

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        ignore_index: int = -1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma       = gamma
        self.ignore_index = ignore_index
        self.reduction   = reduction
        self.alpha       = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # standard CE
        ce = F.cross_entropy(
            logits, targets,
            weight=self.alpha.to(logits.device) if self.alpha is not None else None,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        pt    = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce

        # mask out ignored
        mask  = (targets != self.ignore_index)
        focal = focal[mask]

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


# -----------------------------------------------------------------------------#
#  PCA Utility
# -----------------------------------------------------------------------------#
def apply_pca(X: np.ndarray, num_components: int) -> np.ndarray:
    """
    Applies PCA to a hyperspectral data cube.

    Args:
        X (np.ndarray): The input hyperspectral cube with shape (H, W, D).
        num_components (int): The number of principal components to keep.

    Returns:
        np.ndarray: The dimensionally-reduced data cube with shape (H, W, num_components).
    """
    h, w, d = X.shape
    X_reshaped = np.reshape(X, (-1, d))
    pca = PCA(n_components=num_components, whiten=True)
    X_pca = pca.fit_transform(X_reshaped)
    X_pca_reshaped = np.reshape(X_pca, (h, w, num_components))
    return X_pca_reshaped

def fit_pca_from_pixels(cube: np.ndarray, coords: np.ndarray, num_components: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit PCA on spectral vectors sampled at pixel coords (train split).
    Returns (mean, components) suitable for reuse.
    """
    rows, cols = coords[:, 0], coords[:, 1]
    X = cube[rows, cols, :]                    # (N_train, D)
    pca = PCA(n_components=num_components, whiten=True)
    pca.fit(X)
    return pca.mean_.astype(np.float32), pca.components_.astype(np.float32)

def apply_pca_with_state(cube: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """
    Apply a pre-fitted PCA (mean, components) to a full cube (H, W, D).
    """
    H, W, D = cube.shape
    X = cube.reshape(-1, D)
    Xc = X - mean[None, :]
    Xp = np.dot(Xc, components.T)
    return Xp.reshape(H, W, components.shape[0])
