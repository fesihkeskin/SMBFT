"""Plot ground-truth vs prediction maps for paper figures."""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from typing import Optional

# Set project root and import custom dataset utilities
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import load_dataset, DATASET_NAME_LIST, DATASET_PATHS
from src.data.dataset_info import get_dataset_labels

def harmonize_labels(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensures both GT and prediction arrays have matching label indices.
    Converts -1 in pred to 0 for background/undefined.
    Returns: (gt_harmonized, pred_harmonized)
    """
    gt_harmonized = gt.copy()
    pred_harmonized = pred.copy()

    # Convert -1 (ignore) in pred to 0 (background/undefined)
    pred_harmonized[pred_harmonized < 0] = 0

    gt_labels = np.unique(gt_harmonized)
    pred_labels = np.unique(pred_harmonized)
    if not np.array_equal(gt_labels, pred_labels):
        print(f"[Warning] Label IDs do not fully match: GT {gt_labels}, Pred {pred_labels}")
        # Optionally, raise an error or handle mismatches here

    return gt_harmonized, pred_harmonized

def plot_gt_vs_pred_side_by_side(
    gt: np.ndarray,
    pred: np.ndarray,
    label_names: list,
    dataset_name: str = "",
    save_path: Optional[str] = None,
):
    """
    Visualizes ground truth and prediction maps side by side (or stacked vertically if the shape is very wide),
    with a shared color legend (colorbar). Colors are always consistent between GT and prediction.
    """
    n_classes = len(label_names)
    # Create a consistent discrete colormap
    if n_classes - 1 <= 20:
        base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        colors = ["black"] + [base_colors[i] for i in range(n_classes - 1)]
    else:
        cmap_discrete = plt.cm.get_cmap("tab20", n_classes - 1)
        colors = ["black"] + [cmap_discrete(i) for i in range(n_classes - 1)]
    cmap = ListedColormap(colors)
    norm = mpl.colors.Normalize(vmin=0, vmax=n_classes - 1)

    # Visualization: same mapping for both GT and prediction
    gt_vis = gt.copy()
    pred_vis = pred.copy()
    gt_vis[gt_vis < 0] = 0
    pred_vis[pred_vis < 0] = 0

    height, width = gt.shape[:2]
    scale = 0.05
    aspect_ratio = width / height

    if aspect_ratio >= 1.5:
        # Stack vertically if image is much wider than tall
        layout = 'vertical'
        figsize = (width * scale, height * scale * 2.3)
        fig, axes = plt.subplots(3, 1, figsize=figsize,
                                 gridspec_kw={'height_ratios':[1, 1, 0.07]})
    else:
        # Default: side by side
        layout = 'horizontal'
        figsize = (width * scale * 2.6, height * scale)
        fig, axes = plt.subplots(1, 3, figsize=figsize,
                                 gridspec_kw={'width_ratios':[1, 1, 0.07]})

    base_font_scale = max(figsize)
    TITLE_SIZE = max(10, base_font_scale * 2)
    CBAR_TICK_SIZE = max(8, base_font_scale * 1.5)
    CBAR_LABEL_SIZE = max(9, base_font_scale * 1.5)

    if layout == 'horizontal':
        im0 = axes[0].imshow(gt_vis, cmap=cmap, norm=norm, aspect='equal')
        axes[0].set_title("GT", fontsize=TITLE_SIZE)
        axes[0].axis("off")

        im1 = axes[1].imshow(pred_vis, cmap=cmap, norm=norm, aspect='equal')
        axes[1].set_title("Pred.", fontsize=TITLE_SIZE)
        axes[1].axis("off")

        cbar = plt.colorbar(im1, cax=axes[2], ticks=np.arange(n_classes))
        cbar.ax.set_yticklabels(label_names, fontsize=CBAR_TICK_SIZE)
        cbar.set_label("Classes", rotation=270, labelpad=15, fontsize=CBAR_LABEL_SIZE)
        axes[2].set_frame_on(False)
        plt.subplots_adjust(wspace=0, hspace=0, left=0.02, right=0.8, top=0.96, bottom=0.01)
    else:
        im0 = axes[0].imshow(gt_vis, cmap=cmap, norm=norm, aspect='equal')
        axes[0].axis("off")
        axes[0].text(
            -0.04, 0.5, "GT",
            fontsize=TITLE_SIZE,
            va='center', ha='center',
            rotation=0,
            transform=axes[0].transAxes
        )
        
        im1 = axes[1].imshow(pred_vis, cmap=cmap, norm=norm, aspect='equal')
        axes[1].axis("off")
        axes[1].text(
            -0.04, 0.5, "Pred.",
            fontsize=TITLE_SIZE,
            va='center', ha='center',
            rotation=0,
            transform=axes[1].transAxes
        )

        cbar = plt.colorbar(im1, cax=axes[2], orientation='horizontal', ticks=np.arange(n_classes))
        cbar.ax.set_xticklabels(label_names, fontsize=CBAR_TICK_SIZE, rotation=45, ha='right')
        cbar.set_label("Classes", fontsize=CBAR_LABEL_SIZE)
        axes[2].set_frame_on(False)
        plt.subplots_adjust(wspace=0, hspace=0, left=0.10, right=0.99, top=0.96, bottom=0.06)

    # ... fig, axes = plt.subplots(...) ...
    fig_width, fig_height = fig.get_size_inches()
    dpi = fig.dpi
    pixel_width, pixel_height = int(fig_width * dpi), int(fig_height * dpi)
    print(f"Figure size (inches): {fig_width:.2f} x {fig_height:.2f}  |  Figure size (pixels): {pixel_width} x {pixel_height}")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

def plot_difference_map(
    gt: np.ndarray,
    pred: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    Generates and saves a binary image showing the difference between GT and prediction.
    - White pixels: Incorrect prediction (where GT is not background).
    - Black pixels: Correct prediction or background.
    """
    # Ensure pred has the same shape as gt
    if pred.shape != gt.shape:
        pred = pred.reshape(gt.shape)

    # Create a boolean mask for non-background pixels in the ground truth
    valid_pixels_mask = gt > 0

    # Initialize difference map as all black (0)
    difference_map = np.zeros_like(gt, dtype=np.uint8)

    # Set pixels to white (1) where the prediction is incorrect on valid (non-background) pixels
    incorrect_predictions = (gt[valid_pixels_mask] != pred[valid_pixels_mask])
    difference_map[valid_pixels_mask] = incorrect_predictions.astype(np.uint8)

    # Plotting
    height, width = gt.shape[:2]
    scale = 0.05
    figsize = (width * scale, height * scale)

    fig, ax = plt.subplots(figsize=figsize)
    # Use 'gray' colormap where 0 is black and 1 is white
    ax.imshow(difference_map, cmap='gray', interpolation='none')
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        print(f"✅ Difference map saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

# -- Main script starts here --
SEED = 42  # For reproducibility
DATASET_NAME_LIST = ['Houston13', 'Pavia_University', 'Salinas']
# DATASET_NAME_LIST = ['Salinas'] # For testing purposes, limit to one dataset

for dname in DATASET_NAME_LIST:
    print(f"🔍 Loading dataset: {dname}")
    # Load the dataset using your custom loader
    cube, gt = load_dataset(dname)
    label_names = get_dataset_labels(dname)

    # Load prediction
    smbft_pred_path = Path(PROJECT_ROOT) / "models/final/smbft" / dname / "test_results" / f"pred_map_seed_{SEED}.npy"
    if not smbft_pred_path.exists():
        print(f"[Error] Prediction file not found: {smbft_pred_path}")
        continue
    smbft_pred = np.load(smbft_pred_path)
    print(f"✅ Prediction loaded: {smbft_pred_path}")

    save_path = Path(PROJECT_ROOT) / "reports/results/" / f"{dname}_gt_vs_pred.png"
    print(f"📂 Saving visualization to: {save_path}")

    # Harmonize label indices between GT and pred
    gt_harmonized, smbft_pred_harmonized = harmonize_labels(gt, smbft_pred)

    # Visualize
    plot_gt_vs_pred_side_by_side(
        gt=gt_harmonized,
        pred=smbft_pred_harmonized.reshape(gt.shape),
        label_names=label_names,
        dataset_name=dname,
        save_path=save_path
    )
    print(f"✅ Visualization saved: {save_path}")

    # Generate and save the difference map
    diff_save_path = Path(PROJECT_ROOT) / "reports/results/" / f"{dname}_difference_map.png"
    plot_difference_map(
        gt=gt_harmonized,
        pred=smbft_pred_harmonized,
        save_path=diff_save_path
    )
    print("-" * 30)


print("🎉 All visualizations completed successfully!")