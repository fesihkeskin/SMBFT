"""Visualization helpers for dataset analysis and result plots."""

# src/utils/visualization.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import seaborn as sns
import os
import matplotlib.patches as mpatches
import matplotlib as mpl
from typing import List, Tuple, Optional

# -------------- General Helper -----------------

def save_figure(fig, save_path: Optional[os.PathLike], filename: str, transparent: bool = False):
    """Save a matplotlib figure to the specified directory, with optional transparency and bbox bug workaround."""
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, filename)
        try:
            # Try tight layout (best for most figures)
            fig.savefig(save_file, bbox_inches='tight', transparent=transparent)
        except ValueError as e:
            # If "bboxes cannot be empty" error, save without bbox_inches
            if "'bboxes' cannot be empty" in str(e):
                fig.savefig(save_file, transparent=transparent)
            else:
                raise
    plt.close(fig)

# -------------- Confusion Matrix ----------------

def plot_confusion_matrix(conf_matrix: np.ndarray, class_names: Optional[List[str]] = None,
                         figsize: Tuple[int, int] = (10, 8), cmap: str = 'Blues', title: str = 'Confusion Matrix',
                         save_path: Optional[os.PathLike] = None):
    """
    Plot a confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt="d", 
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.tight_layout()
    save_figure(fig, save_path, "confusion_matrix.png")

def plot_confusion_matrix_from_file(cm_path, class_names=None, save_path=None):
    cm = np.load(cm_path)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

# -------------- Per-Class Accuracy Bar Plot ----------------

def plot_class_accuracy(class_accuracies: List[float], class_names: Optional[List[str]] = None,
                       figsize: Tuple[int, int] = (10, 6), color: str = 'skyblue', title: str = 'Per-Class Accuracy',
                       save_path: Optional[os.PathLike] = None):
    """
    Plot per-class accuracy as a bar chart.
    """
    fig, ax = plt.subplots(figsize=figsize)
    classes = np.arange(len(class_accuracies))
    if class_names is None:
        class_names = [f'Class {i}' for i in classes]
    ax.bar(classes, class_accuracies, color=color)
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_xticks(classes)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    plt.tight_layout()
    save_figure(fig, save_path, "class_accuracy.png")

def plot_per_class_accuracy_from_file(acc_path, class_names=None, save_path=None):
    acc = np.load(acc_path)
    plt.figure(figsize=(8,4))
    plt.bar(np.arange(len(acc)), acc, tick_label=class_names)
    plt.ylim(0,1)
    plt.title("Per-Class Accuracy")
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

# -------------- Class Distribution ----------------

def plot_class_distribution_graph(ground_truth: np.ndarray, dataset_name: str, save_path: Optional[os.PathLike] = None):
    unique_labels, counts = np.unique(ground_truth, return_counts=True)
    labels, cnts = unique_labels[1:], counts[1:]  # Exclude 'Undefined'
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(labels, cnts, color="skyblue")
    ax.set(title=f"Class Distribution - {dataset_name}", xlabel="Class label", ylabel="Pixel count")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    save_figure(fig, save_path, f"class_distribution_{dataset_name}.png")

def plot_class_distribution_table(ground_truth: np.ndarray, label_names: List[str], dataset_name: str, save_path: Optional[os.PathLike] = None):
    unique_labels, counts = np.unique(ground_truth, return_counts=True)
    data = []
    for l, c in zip(unique_labels, counts):
        label_id = int(l)
        count = int(c)
        label_name = label_names[label_id] if label_id < len(label_names) else f"Unknown ({label_id})"
        data.append([label_id, label_name, count])
    fig, ax = plt.subplots(figsize=(6, len(data) * 0.4))
    ax.axis("tight"); ax.axis("off")
    ax.set_title(f"Class Distribution - {dataset_name}")
    tbl = ax.table(cellText=data, colLabels=["#", "Class", "Pixels"], cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(3)))
    plt.tight_layout()
    save_figure(fig, save_path, f"class_table_{dataset_name}.png")

# -------------- Ground Truth & RGB ----------------

def visualize_sample_band(cube: np.ndarray, band: int, dataset_name: str, save_path: Optional[os.PathLike] = None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cube[:, :, band], cmap="gray")
    ax.set_title(f"{dataset_name} - Band {band}")
    ax.axis("off")
    plt.tight_layout()
    save_figure(fig, save_path, f"band_{band}_{dataset_name}.png")

def visualize_ground_truth(gt: np.ndarray, label_names: List[str], dataset_name: str, save_path: Optional[os.PathLike] = None):
    n_classes = len(label_names)
    if n_classes - 1 <= 20:
        base = plt.cm.tab20(np.linspace(0, 1, 20))
        colors = ["black"] + [base[i] for i in range(n_classes - 1)]
    else:
        cmap_discrete = plt.cm.get_cmap("tab20", n_classes - 1)
        colors = ["black"] + [cmap_discrete(i) for i in range(n_classes - 1)]
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(gt, cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax, ticks=np.arange(n_classes))
    cbar.ax.set_yticklabels(label_names)
    cbar.set_label("Classes", rotation=270, labelpad=15)
    cbar.ax.invert_yaxis()
    ax.set_title(f"Ground Truth - {dataset_name}")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path, f"ground_truth_{dataset_name}.png")
    else:
        plt.show()
        
def visualize_rgb_composite(cube: np.ndarray, bands: Tuple[int, int, int], dataset_name: str, save_path: Optional[os.PathLike] = None):
    rgb = np.stack([cube[:, :, b] for b in bands], axis=-1)
    rgb_min = rgb.min(axis=(0, 1), keepdims=True)
    rgb_max = rgb.max(axis=(0, 1), keepdims=True)
    rgb_norm = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb_norm)
    ax.set_title(f"{dataset_name} - RGB composite bands {bands}")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path, f"rgb_composite_{dataset_name}_{bands}.png")
    else:
        plt.show()
        
def plot_spectral_signature(cube: np.ndarray, coord: Tuple[int, int], dataset_name: str, save_path: Optional[os.PathLike] = None):
    row, col = coord
    spectral_curve = cube[row, col, :]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(spectral_curve.shape[0]), spectral_curve, marker="o", lw=1)
    ax.set_title(f"{dataset_name} - Spectral signature at pixel ({row}, {col})")
    ax.set_xlabel("Spectral Band")
    ax.set_ylabel("Reflectance")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path, f"spectral_signature_{dataset_name}_{row}_{col}.png")
    else:
        plt.show()
        
def plot_rgb_and_ground_truth_side_by_side(
    cube: np.ndarray,
    gt: np.ndarray,
    label_names: List[str],
    bands: Tuple[int, int, int],
    dataset_name: str,
    save_path: Optional[os.PathLike] = None,
):
    rgb = np.stack([cube[:, :, b] for b in bands], axis=-1)
    rgb_min = rgb.min(axis=(0, 1), keepdims=True)
    rgb_max = rgb.max(axis=(0, 1), keepdims=True)
    rgb_norm = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)
    n_classes = len(label_names)
    if n_classes - 1 <= 20:
        base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        colors = ["black"] + [base_colors[i] for i in range(n_classes - 1)]
    else:
        cmap_discrete = plt.cm.get_cmap("tab20", n_classes - 1)
        colors = ["black"] + [cmap_discrete(i) for i in range(n_classes - 1)]
    cmap = ListedColormap(colors)
    height, width = cube.shape[:2]
    scale = 0.05
    figsize = (width * scale * 2, height * scale)
    base_font_scale = max(figsize)
    TITLE_SIZE = max(10, base_font_scale * 2)
    CBAR_TICK_SIZE = max(8, base_font_scale * 1.5)
    CBAR_LABEL_SIZE = max(9, base_font_scale * 2)
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    axes[0].imshow(rgb_norm)
    axes[0].set_title("RGB Composite", fontsize=TITLE_SIZE)
    axes[0].axis("off")
    im = axes[1].imshow(gt, cmap=cmap)
    axes[1].set_title(f"GT", fontsize=TITLE_SIZE)
    axes[1].axis("off")
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax, ticks=np.arange(n_classes))
    cbar.ax.set_yticklabels(label_names, fontsize=CBAR_TICK_SIZE)
    cbar.set_label("Classes", rotation=270, labelpad=15, fontsize=CBAR_LABEL_SIZE)
    cbar.ax.invert_yaxis()
    if save_path:
        save_figure(fig, save_path, f"rgb_and_gt_{dataset_name}_{bands}.png")
    else:
        plt.show()
        
# -------------- 3D Hyperspectral Cube Plot ----------------

def plot_3d_hyperspectral_cube(
    cube: np.ndarray,
    rgb_band_indices: Tuple[int, int, int],
    dataset_name: str,
    save_path: Optional[os.PathLike] = None,
    interactive: bool = False,
    grid_off: bool = False,
    no_axes: bool = False,
):
    n_rows, n_cols, n_bands = cube.shape
    rgb_image = np.stack([cube[:, :, band] for band in rgb_band_indices], axis=-1)
    rgb_min = rgb_image.min(axis=(0, 1), keepdims=True)
    rgb_max = rgb_image.max(axis=(0, 1), keepdims=True)
    rgb_norm = (rgb_image - rgb_min) / (rgb_max - rgb_min + 1e-8)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    ax.plot_surface(
        X, Y, np.full_like(X, n_bands - 1),
        rstride=1, cstride=1,
        facecolors=rgb_norm[:, ::-1],
        shade=False
    )
    last_col_spectrum = cube[:, -1, :]
    side_norm = (last_col_spectrum - last_col_spectrum.min()) / (last_col_spectrum.max() - last_col_spectrum.min() + 1e-8)
    X_side, Y_side = np.meshgrid(np.arange(n_bands), np.arange(n_rows))
    ax.plot_surface(
        np.full_like(X_side, n_cols - 1), Y_side, X_side,
        facecolors=plt.cm.viridis(side_norm),
        rstride=1, cstride=1, shade=False
    )
    last_row_spectrum = cube[-1, :, :]
    top_norm = (last_row_spectrum - last_row_spectrum.min()) / (last_row_spectrum.max() - last_row_spectrum.min() + 1e-8)
    Z_top, X_top = np.meshgrid(np.arange(n_bands), np.arange(n_cols))
    ax.plot_surface(
        X_top, np.full_like(X_top, n_rows - 1), Z_top,
        facecolors=plt.cm.viridis(top_norm),
        rstride=1, cstride=1, shade=False
    )
    first_col_spectrum = cube[:, 0, :]
    side_norm_left = (first_col_spectrum - first_col_spectrum.min()) / (first_col_spectrum.max() - first_col_spectrum.min() + 1e-8)
    X_side_left, Y_side_left = np.meshgrid(np.arange(n_bands), np.arange(n_rows))
    ax.plot_surface(
        np.full_like(X_side_left, 0), Y_side_left, X_side_left,
        facecolors=plt.cm.viridis(side_norm_left),
        rstride=1, cstride=1, shade=False
    )
    first_row_spectrum = cube[0, :, :]
    top_norm_bottom = (first_row_spectrum - first_row_spectrum.min()) / (first_row_spectrum.max() - first_row_spectrum.min() + 1e-8)
    Z_bottom, X_bottom = np.meshgrid(np.arange(n_bands), np.arange(n_cols))
    ax.plot_surface(
        X_bottom, np.full_like(X_bottom, 0), Z_bottom,
        facecolors=plt.cm.viridis(top_norm_bottom),
        rstride=1, cstride=1, shade=False
    )
    if not no_axes:
        ax.set_xlabel('Width (columns)')
        ax.set_ylabel('Height (rows)')
        ax.set_zlabel('Spectral Bands')
        ax.set_title(f"3D Hyperspectral Cube - {dataset_name}")
    else:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
        ax.set_title('')
        ax.xaxis.pane.set_visible(False); ax.yaxis.pane.set_visible(False); ax.zaxis.pane.set_visible(False)
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.xaxis.line.set_visible(False)
        ax.yaxis.line.set_visible(False)
        ax.zaxis.line.set_visible(False)
    if grid_off:
        ax.grid(False)
    ax.view_init(elev=60, azim=150)
    if interactive:
        def on_release(event):
            if event.inaxes is ax:
                print(f"[VIEW] elev={ax.elev:.1f}, azim={ax.azim:.1f}")
        def on_close(event):
            print(f"[FINAL VIEW] elev={ax.elev:.1f}, azim={ax.azim:.1f}")
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('close_event', on_close)
        plt.show()
    else:
        save_figure(fig, save_path, f"3d_cube_{dataset_name}.png", transparent=True)

# -------------- Save Metrics Utility ----------------

def save_metrics_to_file(metrics_dict, filename='classification_metrics.txt'):
    """
    Save classification metrics to a text file.
    """
    with open(filename, 'w') as f:
        f.write(f"Overall Accuracy (OA): {metrics_dict['oa']:.4f}\n")
        f.write(f"Average Accuracy (AA): {metrics_dict['aa']:.4f}\n")
        f.write(f"Kappa Coefficient: {metrics_dict['kappa']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(metrics_dict['conf_matrix']))
        f.write("\n\nClassification Report:\n")
        f.write(metrics_dict['cls_report'])

def plot_gt_vs_pred_side_by_side(
    gt: np.ndarray,
    pred: np.ndarray,
    label_names: list,
    dataset_name: str = "",
    save_path: Optional[os.PathLike] = None,
):
    """
    Ground truth ve prediction haritalarını yan yana ve renkli legend/cbar ile gösterir.
    Class sayısına ve görüntü yüksekliğine göre otomatik boyut ve font ayarlar.
    """
    n_classes = len(label_names)
    # Color map ayarı
    if n_classes - 1 <= 20:
        base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        colors = ["black"] + [base_colors[i] for i in range(n_classes - 1)]
    else:
        cmap_discrete = plt.cm.get_cmap("tab20", n_classes - 1)
        colors = ["black"] + [cmap_discrete(i) for i in range(n_classes - 1)]
    cmap = ListedColormap(colors)
    norm = mpl.colors.Normalize(vmin=0, vmax=n_classes - 1)

    # Prediction'da -1'leri (ignore) ve 0'ları (background) düzgün göster
    gt_vis = gt.copy()
    pred_vis = pred.copy()
    pred_vis[pred_vis < 0] = 0

    # Otomatik figsize ve font boyutları
    height, width = gt.shape[:2]
    scale = 0.05
    figsize = (width * scale * 2, height * scale)
    base_font_scale = max(figsize)
    TITLE_SIZE = max(10, base_font_scale * 0.5)
    CBAR_TICK_SIZE = max(8, base_font_scale * 0.5)
    CBAR_LABEL_SIZE = max(9, base_font_scale * 0.8)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    im0 = axes[0].imshow(gt_vis, cmap=cmap, norm=norm)
    axes[0].set_title("GT", fontsize=TITLE_SIZE)
    axes[0].axis("off")
    im1 = axes[1].imshow(pred_vis, cmap=cmap, norm=norm)
    axes[1].set_title("Pred.", fontsize=TITLE_SIZE)
    axes[1].axis("off")
    # subplotlar arası boşlukları azalt
    plt.subplots_adjust(wspace=0.02, left=0.01, right=0.98, top=0.90, bottom=0.01)
    # Colorbar (legend gibi)
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im1, cax=cax, ticks=np.arange(n_classes))
    cbar.ax.set_yticklabels(label_names, fontsize=CBAR_TICK_SIZE)
    cbar.set_label("Classes", rotation=270, labelpad=15, fontsize=CBAR_LABEL_SIZE)
    cbar.ax.invert_yaxis()

    # Kayıt
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()