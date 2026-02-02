"""Dataset metadata, label maps, and visualization helpers."""

# src/data/dataset_info.py

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence, Tuple, Optional

import numpy as np
import scipy.io

# Make sure the project root is in the path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Public API (exported symbols)
__all__: Sequence[str] = [
    "get_dataset_labels",
    "dataset_summary",
    "subsample_cube",
    "remap_rgb_bands",
]


# Dataset label maps
def get_dataset_labels(dataset_name: str) -> list[str]:
    """Return the class label names for the specified dataset."""
    datasets = {
        "Houston13": [
            "Undefined", "Healthy grass", "Stressed grass", "Synthetic grass", "Trees",
            "Soil", "Water", "Residential", "Commercial", "Road", "Highway",
            "Railway", "Parking Lot 1", "Parking Lot 2", "Tennis Court", "Running Track"
        ],
        "Pavia_University": [
            "Undefined", "Asphalt", "Meadows", "Gravel", "Trees",
            "Painted metal sheets", "Bare Soil", "Bitumen",
            "Self-Blocking Bricks", "Shadows",
        ],
        "Salinas": [
            "Undefined", "Broccoli_green_weeds_1", "Broccoli_green_weeds_2",
            "Fallow", "Fallow_rough_plow", "Fallow_smooth", "Stubble",
            "Celery", "Grapes_untrained", "Soil_vineyard_develop",
            "Corn_senesced_green_weeds", "Lettuce_romaine_4wk",
            "Lettuce_romaine_5wk", "Lettuce_romaine_6wk",
            "Lettuce_romaine_7wk", "Vineyard_untrained",
            "Vineyard_vertical_trellis",
        ],
    }
    if dataset_name not in datasets:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")
    return datasets[dataset_name]

# Convenience mapping for dataset names to human-readable names
DATASET_NAME_MAP = {
    "Houston13": "Houston 2013",
    "Pavia_University": "Pavia University",
    "Salinas": "Salinas",
}


# Dataset summary
def dataset_summary(image: np.ndarray, ground_truth: np.ndarray) -> None:
    """
    Print a summary of the dataset image cube and ground truth mask.
    """
    print("Dataset Summary:")
    print(f"- Image Shape: {image.shape} (rows x cols x bands)")
    print(f"- Image Data Type: {image.dtype}")
    print(f"- Number of Spectral Bands: {image.shape[2]}")
    print(f"- Image Value Range: {image.min()} : {image.max()}")

    unique_labels, counts = np.unique(ground_truth, return_counts=True)
    int_labels = [int(l) for l in unique_labels]
    int_counts = [int(c) for c in counts]

    print("\nGround Truth (Labels) Summary:")
    print(f"- Labels Shape: {ground_truth.shape} (rows x cols)")
    print(f"- Labels Data Type: {ground_truth.dtype}")
    print(f"- Number of Classes: {len(unique_labels) - 1}")
    print(f"- Class Distribution: {dict(zip(int_labels[1:], int_counts[1:]))}\n")


# Patch/utility functions
def subsample_cube(
    cube: np.ndarray,
    max_rows: int,
    max_cols: int,
    max_bands: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Subsample the hyperspectral cube to fit given max dimensions.
    Returns:
        subsampled_cube, row_indices, col_indices, band_indices
    """
    n_rows, n_cols, n_bands = cube.shape
    row_indices = np.linspace(0, n_rows - 1, min(n_rows, max_rows), dtype=int)
    col_indices = np.linspace(0, n_cols - 1, min(n_cols, max_cols), dtype=int)
    band_indices = np.linspace(0, n_bands - 1, min(n_bands, max_bands), dtype=int)
    return cube[np.ix_(row_indices, col_indices, band_indices)], row_indices, col_indices, band_indices

def remap_rgb_bands(
    rgb_band_indices: Tuple[int, int, int],
    band_indices: np.ndarray
) -> Tuple[int, int, int]:
    """
    Map original RGB band indices to indices in the subsampled band axis.
    """
    remapped = []
    for band in rgb_band_indices:
        closest_idx = np.argmin(np.abs(band_indices - band))
        remapped.append(int(closest_idx))
    return tuple(remapped)

def remap_labels_if_needed(gt_path, out_path):
    gt = scipy.io.loadmat(gt_path)
    key = [k for k in gt.keys() if not k.startswith("__")][0]
    arr = gt[key]
    unique_labels = np.unique(arr)
    if np.array_equal(unique_labels, np.arange(unique_labels.min(), unique_labels.max() + 1)):
        print(f"✅ {gt_path}: Labels are already consecutive. No remapping needed.")
        return
    lut = np.zeros(unique_labels.max() + 1, dtype=int)
    lut[unique_labels] = np.arange(len(unique_labels))
    arr_remap = lut[arr]
    print(f"{gt_path}: Original labels:", unique_labels)
    print(f"{gt_path}: Remapped to:", np.arange(len(unique_labels)))
    scipy.io.savemat(out_path, {f"{key}_remap": arr_remap})
    print(f"⚠️ Remapped GT saved to {out_path}")

def get_imbalance_ratio(gt: np.ndarray) -> float:
    """
    Calculate the imbalance ratio (max/min) for either:
      • 2D 1-indexed GT with 0 as background, or
      • 1D label vector with labels in [0..C-1] and -1 as ignore.
    """
    arr = gt
    if arr.ndim == 2:
        # 1-indexed GT matrix (0 = background): drop background
        labels = arr[arr > 0]
    else:
        # Vector labels (0..C-1), with -1 for ignore
        labels = arr[arr >= 0]
    unique, counts = np.unique(labels, return_counts=True)
    if counts.size == 0:
        return 1.0
    return float(counts.max() / counts.min())

# CLI Example Usage (for quick data exploration)
if __name__ == "__main__":
    # Import visualization functions from utils
    from src.data.data_loader import load_dataset, DATASET_NAME_LIST, DATASET_PATHS
    from src.utils.visualization import (
        plot_class_distribution_graph,
        plot_class_distribution_table,
        visualize_sample_band,
        visualize_ground_truth,
        visualize_rgb_composite,
        plot_spectral_signature,
        plot_rgb_and_ground_truth_side_by_side,
        plot_3d_hyperspectral_cube,
    )

    for name, paths in DATASET_PATHS.items():
        gt_path = paths["ground_truth"]
        out_path = gt_path.parent / (gt_path.stem + ".mat")
        remap_labels_if_needed(gt_path, out_path)
        
    SAVE_DIR = Path("reports/figures")
    BAND_TO_VISUALIZE = 15
    PIXEL_COORD = (50, 50)

    # Dataset-specific RGB band indices
    RGB_BAND_MAP = {
        "Houston13": (30, 20, 10),
        "Pavia_University": (60, 30, 2),
        "Salinas": (30, 20, 10),
    }
    

    
    # If you want to visualize specific datasets, you can modify list
    DATASET_NAME_LIST = ["Houston13", "Pavia_University", "Salinas"]
    for dname in DATASET_NAME_LIST:
        print(f"🔍 Loading dataset: {dname}")

        cube, gt = load_dataset(dname)
        labels = get_dataset_labels(dname)
        imbalance_ratio = get_imbalance_ratio(gt)
        print(f"⚖️ Imbalance ratio for {dname}: {imbalance_ratio:.2f}")

        # If you want to visualize a specific patch, modify the section below
        draw_sub_cube = False  # Set to True to visualize only a patch
        if draw_sub_cube:
            r, c = 100, 50  # Patch center
            k = 200         # Patch size (k x k)
            b = 10          # Number of bands
            row_start = max(r - k//2, 0)
            row_end = min(r + k//2 + 1, cube.shape[0])
            col_start = max(c - k//2, 0)
            col_end = min(c + k//2 + 1, cube.shape[1])
            cube = cube[row_start:row_end, col_start:col_end, :b]

        rgb_bands = RGB_BAND_MAP.get(dname, (30, 20, 10))

        print(f"\n🗃️ {dname} — Ignored class id: {labels.index('Undefined')}")
        dataset_summary(cube, gt)

        # Standard visualizations
        # plot_class_distribution_graph(gt, dname, SAVE_DIR)
        # plot_class_distribution_table(gt, labels, dname, SAVE_DIR)
        # visualize_sample_band(cube, BAND_TO_VISUALIZE, dname, SAVE_DIR)
        # visualize_ground_truth(gt, labels, dname, SAVE_DIR)
        # visualize_rgb_composite(cube, rgb_bands, dname, SAVE_DIR)
        # plot_spectral_signature(cube, PIXEL_COORD, dname, SAVE_DIR)
        # plot_rgb_and_ground_truth_side_by_side(cube, gt, labels, rgb_bands, dname, SAVE_DIR)

        # 3D cube visualization (subsampling for performance if needed)
        subsample_rate = 1
        cube_small, row_idx, col_idx, band_idx = subsample_cube(
            cube,
            max_rows=int(cube.shape[0] / subsample_rate),
            max_cols=int(cube.shape[1] / subsample_rate),
            max_bands=int(cube.shape[2] / subsample_rate)
        )
        rgb_bands_sub = remap_rgb_bands(rgb_bands, band_idx)
        plot_3d_hyperspectral_cube(
            cube_small,
            rgb_bands_sub,
            dname,
            save_path=SAVE_DIR, # if interactive=True, this will not save the plot
            interactive=False, # True to enable interactive rotation
            grid_off=True,
            no_axes=True
        )

    print(f"\n✅ All visualizations saved in: {SAVE_DIR.resolve()}")