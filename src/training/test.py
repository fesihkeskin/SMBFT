#!/usr/bin/env python3
"""Evaluate SMBFT checkpoints and save test metrics/plots."""

# src/training/test.py

import os
import json
from pathlib import Path
import matplotlib as mpl
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)

import torch
from torch.utils.data import DataLoader
import matplotlib.patches as mpatches
from argparse import Namespace

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Project imports
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import HyperspectralDataset
from src.data.data_loader import DATASET_PATHS, load_dataset
from src.models.model_architecture import SMBFT
from src.utils.utils import set_seed, apply_pca, apply_pca_with_state
from src.data.dataset_info import get_dataset_labels
from src.utils.visualization import plot_gt_vs_pred_side_by_side

def _build_model(args: Namespace, in_ch: int, n_cls: int):
    """Builds the SMBFT model using arguments from a loaded checkpoint."""
    return SMBFT(
        in_bands=in_ch,
        patch_size=args.patch_size,
        n_classes=n_cls,
        spec_dim=args.spec_dim,
        spat_dim=args.spat_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        fusion=args.fusion,
        mask_ratio=args.mask_ratio,
        mask_mode=args.mask_mode,
        dim_feedforward=args.dim_feedforward,
        dropout=getattr(args, "dropout", 0.1), # Default for older models
        d3_out_channels=getattr(args, "d3_out_channels", 16),
        use_cross_attn=not getattr(args, "no_cross_attn", False), # Default to True
    )


def test(args: Namespace):
    """Main evaluation routine with result saving."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 device:", device)

    # --- LOAD CHECKPOINT and Training Arguments ---
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "args" not in ckpt:
        raise KeyError("Checkpoint is missing 'args' dict. Please retrain the model to include it.")

    # Load model arguments from the checkpoint for consistency
    model_args = Namespace(**ckpt['args'])
    
    # Use the seed from the training run for reproducibility
    set_seed(model_args.seed)

    # --- Load data arrays & resolve file paths based on CHECKPOINT ARGS ---
    img_arr, gt_arr = load_dataset(model_args.dataset)

    # Apply PCA using saved state (preferred), else legacy fallback
    if getattr(model_args, "pca_components", 0) and model_args.pca_components > 0:
        if hasattr(model_args, "pca_state"):
            print(f"Applying PCA with saved state: {model_args.pca_components} components.")
            mean = np.array(model_args.pca_state["mean"], dtype=np.float32)
            comps = np.array(model_args.pca_state["components"], dtype=np.float32)
            # Defensive check for dimension mismatch
            if mean.shape[0] != img_arr.shape[2] or comps.shape[1] != img_arr.shape[2]:
                print(f"⚠️ PCA state mismatch (state D={mean.shape[0]}, data D={img_arr.shape[2]}). Recomputing PCA on full cube (legacy fallback).")
                img_arr = apply_pca(img_arr, num_components=model_args.pca_components)
            else:
                img_arr = apply_pca_with_state(img_arr, mean, comps)
        else:
            print(f"Applying PCA from scratch: {model_args.pca_components} components (legacy).")
            img_arr = apply_pca(img_arr, num_components=model_args.pca_components)
        print(f"✅ PCA complete. New image shape: {img_arr.shape}")

    # Attach data to the args namespace that will be used to build datasets
    model_args.raw_gt = gt_arr
    model_args.raw_cube = img_arr
    model_args.cube_path, model_args.gt_path = (
        PROJECT_ROOT / DATASET_PATHS[model_args.dataset]["image"],
        PROJECT_ROOT / DATASET_PATHS[model_args.dataset]["ground_truth"]
    )

    # --- LOAD SPLIT INDICES based on training's output directory and seed ---
    # The output_dir in model_args points to the correct seed-specific folder
    split_dir = Path(model_args.output_dir) / "splits"
    train_idx_path = split_dir / f"train_idx_seed_{model_args.seed}.npy"
    test_idx_path  = split_dir / f"test_idx_seed_{model_args.seed}.npy"
    if not train_idx_path.exists() or not test_idx_path.exists():
        raise FileNotFoundError(f"Missing split indices in {split_dir}. Expected from training run.")
    train_idx = np.load(train_idx_path)
    test_idx  = np.load(test_idx_path)

    # Build train_ds using the *exact* train_idx so mean/std match
    train_ds = HyperspectralDataset(
        cube_path=model_args.cube_path, gt_path=model_args.gt_path,
        patch_size=model_args.patch_size,
        stride=model_args.stride,
        mode="train",
        indices=train_idx,
        mean=None, std=None, augment=False,
        cfg=model_args,
        preloaded_cube=model_args.raw_cube, # Use the (potentially PCA'd) cube
    )
    
    # Build test_ds with saved test_idx and same normalization
    test_ds = HyperspectralDataset(
        cube_path=model_args.cube_path, gt_path=model_args.gt_path,
        patch_size=model_args.patch_size,
        stride=1,
        mode="test",
        indices=test_idx,
        mean=train_ds.mean, std=train_ds.std, augment=False,
        cfg=model_args,
        preloaded_cube=model_args.raw_cube, # Use the (potentially PCA'd) cube
    )

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size, # Use runtime batch size for testing flexibility
        shuffle=False,
        num_workers=args.num_workers, # Use runtime workers
        pin_memory=(device.type == "cuda"),
    )

    # Build model from checkpoint args and load weights
    n_cls = len(np.unique(model_args.raw_gt)) - 1
    # CRITICAL: Pass the correct number of input bands (train_ds.D)
    model = _build_model(model_args, in_ch=train_ds.D, n_cls=n_cls)
    model.to(device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model.eval()

    # ── 1. Inference + Prediction Map Generation ───────────────────
    y_true, y_pred = [], []
    n_samples = len(test_ds)
    infer_time = 0.0
    infer_time_per_sample = 0.0

    # Create a base map with ground truth for non-test pixels
    pred_map = np.full(test_ds.gt.shape, -1, dtype=int)
    
    # Fill in ground truth for the training split
    for r, c in train_idx:
        pred_map[r, c] = test_ds.gt[r, c]
        
    # Fill in ground truth for the validation split, if it exists
    val_idx_path = split_dir / f"val_idx_seed_{model_args.seed}.npy"
    if val_idx_path.exists():
        val_idx = np.load(val_idx_path)
        for r, c in val_idx:
            pred_map[r, c] = test_ds.gt[r, c]
    
    # Run inference on the test split
    import time
    start_time = time.time()
    for batch in tqdm(loader, desc="Testing"):
        x, labels = batch["patch"].to(device), batch["label"].cpu().numpy()
        coords = batch["coord"].cpu().numpy()
        with torch.no_grad():
            logits = model(x)
        preds = logits.argmax(1).cpu().numpy()

        # Collect ground-truth vs. predictions
        mask = labels != -1
        y_true.extend(labels[mask])
        y_pred.extend(preds[mask])

        # Add test predictions to the map
        for i, (r, c) in enumerate(coords):
            if mask[i]:
                pred_map[r, c] = preds[i] + 1  # +1 to align with 1-based GT labels
    infer_time = time.time() - start_time
    infer_time_per_sample = infer_time / n_samples if n_samples > 0 else float('nan')

    # Remove background (0) and ignored pixels (-1) for metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = (y_true >= 0)
    y_true_filt = y_true[mask]
    y_pred_filt = y_pred[mask]

    # Get labels and class names for reporting
    present_labels = np.unique(y_true_filt)
    full_class_names = get_dataset_labels(model_args.dataset)
    # Map numeric labels to string names for the report
    present_class_names = [full_class_names[i+1] for i in present_labels]

    # Calculate metrics
    cm = confusion_matrix(y_true_filt, y_pred_filt, labels=present_labels)
    per_class_acc = (cm.diagonal() / cm.sum(axis=1)).tolist()
    aa    = float(np.mean(per_class_acc))
    oa    = accuracy_score(y_true_filt, y_pred_filt)
    kappa = cohen_kappa_score(y_true_filt, y_pred_filt)

    # Classification report
    cls_report = classification_report(
        y_true_filt, y_pred_filt,
        labels=present_labels,
        target_names=present_class_names,
        digits=4,
        zero_division=0,
        output_dict=True
    )
    print(classification_report(
        y_true_filt, y_pred_filt,
        labels=present_labels,
        target_names=present_class_names,
        digits=4,
        zero_division=0
    ))
    print(f"\nOA  {oa:.4f}  |  AA  {aa:.4f}  |  κ  {kappa:.4f}\n")

    # Prepare results directory (use the runtime output_dir)
    results_dir = Path(args.output_dir) / "test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics JSON
    metrics = {
        "Overall Accuracy": oa,
        "Average Accuracy": aa,
        "Kappa": kappa,
        "Per Class Accuracy": dict(zip(present_class_names, per_class_acc)),
        "Classification Report": cls_report,
        "Inference Time Per Sample (seconds)": infer_time_per_sample,
        "Inference Time Total (seconds)": infer_time,
        "Number of Samples": n_samples
    }
    with open(results_dir / f"metrics_seed_{model_args.seed}.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Plot confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=present_class_names,
                yticklabels=present_class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(results_dir / f"confusion_matrix_seed_{model_args.seed}.png")
    plt.close()

    # Save confusion matrix as text file
    cm_txt_path = results_dir / f"confusion_matrix_seed_{model_args.seed}.txt"
    np.savetxt(cm_txt_path, cm, fmt="%d", delimiter="\t")

    # Plot per-class accuracy bar chart
    plt.figure(figsize=(10,5))
    plt.bar(np.arange(len(per_class_acc)), per_class_acc, tick_label=present_class_names)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0,1)
    plt.title("Per-Class Accuracy")
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(results_dir / f"per_class_accuracy_seed_{model_args.seed}.png")
    plt.close()
    
    # ── 2. Save Results ───────────────────────────────────────────────
    np.save(results_dir / f"pred_map_seed_{model_args.seed}.npy", pred_map)

    # ── 3. Side-by-side Ground Truth vs. Prediction Plot ──────
    plot_gt_vs_pred_side_by_side(
        gt=test_ds.gt,
        pred=pred_map,
        label_names=full_class_names,
        dataset_name=model_args.dataset,
        save_path=results_dir / f"gt_vs_pred_side_by_side_seed_{model_args.seed}.png"
    )

    # Print summary
    print(f"✅ Test completed. Results saved to: {results_dir.resolve()}")

    # ── 3. Side-by-side Ground Truth vs. Prediction Plot ──────    import matplotlib as mpl
    labels_full = get_dataset_labels(args.dataset)
    n_colors    = len(labels_full)  # tüm label isimleri (arkaplan dahil)

    # shared Normalize ve discrete colormap
    norm = mpl.colors.Normalize(vmin=0, vmax=n_colors-1)
    cmap = mpl.cm.get_cmap("tab20", n_colors)

    # # ground truth ve pred aralığını, dataset label indeksine uyacak şekilde
    # gt_vis = test_ds.gt.copy()            # raw GT: 1..n_colors
    # pred_vis = pred_map.copy()            
    # pred_vis[pred_vis < 0] = 0            # background için 0

    # fig, axes = plt.subplots(1,3, figsize=(18,6))
    # im0 = axes[0].imshow(gt_vis, cmap=cmap, norm=norm)
    # axes[0].set_title("Ground Truth"); axes[0].axis("off")
    # im1 = axes[1].imshow(pred_vis, cmap=cmap, norm=norm)
    # axes[1].set_title("Prediction"); axes[1].axis("off")

    # # # legend
    # axes[2].axis("off")
    # patches = [mpatches.Patch(color=cmap(i), label=labels_full[i]) for i in range(n_colors)]
    # axes[2].legend( handles=patches, title=r"$\bf{Classes}$", bbox_to_anchor=(1.05, 1), borderaxespad=0. )

    # plt.savefig(results_dir / f"gt_vs_pred_seed_{args.seed}.png", dpi=200, bbox_inches='tight', pad_inches=0)
    # plt.close()

    # Save confusion matrix and per-class accuracy for later visualization
    np.save(results_dir / f"confusion_matrix_seed_{args.seed}.npy", cm)
    np.save(results_dir / f"per_class_accuracy_seed_{args.seed}.npy", np.array(per_class_acc))
    np.save(results_dir / f"pred_map_seed_{args.seed}.npy", pred_map)
    with open(results_dir / f"metrics_seed_{args.seed}.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Print summary
    print("Results saved to:", results_dir)
    print(f"Metrics JSON saved to: {results_dir / f'metrics_seed_{args.seed}.json'}")
    print(f"Confusion matrix (npy) saved to: {results_dir / f'confusion_matrix_seed_{args.seed}.npy'}")
    print(f"Per-class accuracy (npy) saved to: {results_dir / f'per_class_accuracy_seed_{args.seed}.npy'}")
    print(f"Ground truth vs prediction image saved to: {results_dir / f'gt_vs_pred_seed_{args.seed}.png'}")
    print(f"Prediction map saved to: {results_dir / f'pred_map_seed_{args.seed}.npy'}")
    print("✅ Test completed successfully.")

    plot_gt_vs_pred_side_by_side(
        gt=test_ds.gt,
        pred=pred_map,
        label_names=full_class_names,
        dataset_name=model_args.dataset,
        save_path=results_dir / f"gt_vs_pred_side_by_side_seed_{model_args.seed}.png"
    )

    # Print summary
    print(f"✅ Test completed. Results saved to: {results_dir.resolve()}")