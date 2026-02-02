#!/usr/bin/env python3
"""Generate t-SNE feature visualizations from SMBFT checkpoints."""

# src/training/tsne_plot.py
# usage: python src/training/tsne_plot.py --dataset Houston13 --checkpoint models/final/smbft/Houston13/smbft_best_Houston13.pth --force

import os
import sys
from pathlib import Path
import argparse

from argparse import Namespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

FIGURE_PATH = PROJECT_ROOT / "reports" / "figures"

from src.data.data_loader import load_dataset, DATASET_PATHS
from src.data.dataset_info import get_dataset_labels, DATASET_NAME_MAP
from src.data.dataset import HyperspectralDataset
from src.models.model_architecture import SMBFT
from src.utils.utils import set_seed, apply_pca, apply_pca_with_state

def extract_features(model, loader, device):
    """Extracts features from the model for the given data loader."""
    model.eval()
    feats_list, labels_list = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            x = batch["patch"].to(device)
            y = batch["label"].cpu().numpy()
            # Use the extract_features flag to get pooled features
            features_batch = model(x, extract_features=True)
            feats_list.append(features_batch.cpu().numpy())
            labels_list.append(y)
    features = np.concatenate(feats_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return features, labels

def get_discrete_hsi_cmap(n_classes):
    """Creates a discrete colormap for HSI ground truth plots."""
    if n_classes - 1 <= 20:
        base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        colors = ["black"] + [base_colors[i] for i in range(n_classes - 1)]
    else:
        cmap_discrete = plt.cm.get_cmap("tab20", n_classes - 1)
        colors = ["black"] + [cmap_discrete(i) for i in range(n_classes - 1)]
    cmap = ListedColormap(colors)
    norm = mpl.colors.Normalize(vmin=0, vmax=n_classes - 1)
    return cmap, norm

def _build_model(args: Namespace, in_ch: int, n_cls: int):
    """Helper to build the SMBFT model from checkpoint arguments."""
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
        dropout=getattr(args, "dropout", 0.1),
        d3_out_channels=getattr(args, "d3_out_channels", 16),
        use_cross_attn=not getattr(args, "no_cross_attn", False),  # default to True
    )

def tsne_main(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    out_path = Path(args.output_dir) / f"tsne_{args.dataset}_seed{args.seed}.png"
    if out_path.exists() and not args.force:
        print(f"t-SNE plot already exists at {out_path}, skipping. Use --force to overwrite.")
        return str(out_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "args" not in ckpt:
        raise KeyError("Checkpoint is missing 'args' dict. Please retrain the model.")
    
    model_args = Namespace(**ckpt['args'])
    print(f"Loaded model arguments from checkpoint trained with seed {model_args.seed}")

    img_arr, gt_arr = load_dataset(model_args.dataset)
    
    # Determine the number of input bands for the model
    if model_args.pca_components > 0:
        if hasattr(model_args, "pca_state"):
            print(f"Applying PCA with saved state: {model_args.pca_components} components.")
            mean = np.array(model_args.pca_state["mean"], dtype=np.float32)
            comps = np.array(model_args.pca_state["components"], dtype=np.float32)
            if mean.shape[0] != img_arr.shape[2] or comps.shape[1] != img_arr.shape[2]:
                print(
                    f"⚠️ PCA state mismatch (state D={mean.shape[0]}, data D={img_arr.shape[2]}). "
                    "Recomputing PCA on full cube."
                )
                img_arr = apply_pca(img_arr, num_components=model_args.pca_components)
            else:
                img_arr = apply_pca_with_state(img_arr, mean, comps)
        else:
            print(f"Applying PCA from scratch: {model_args.pca_components} components.")
            img_arr = apply_pca(img_arr, num_components=model_args.pca_components)
        # The number of input bands for the model is the number of PCA components
        model_in_bands = model_args.pca_components
    else:
        # If no PCA, the number of bands is from the original data cube
        model_in_bands = img_arr.shape[2]

    model_args.raw_cube = img_arr
    model_args.raw_gt = gt_arr
    model_args.cube_path = PROJECT_ROOT / DATASET_PATHS[model_args.dataset]["image"]
    model_args.gt_path = PROJECT_ROOT / DATASET_PATHS[model_args.dataset]["ground_truth"]

    split_dir = Path(model_args.output_dir) / "splits"
    train_idx_path = split_dir / f"train_idx_seed_{model_args.seed}.npy"
    test_idx_path = split_dir / f"test_idx_seed_{model_args.seed}.npy"
    
    if not train_idx_path.exists() or not test_idx_path.exists():
        raise FileNotFoundError(f"Split indices not found in {split_dir}. Run training first.")
        
    train_coords = np.load(train_idx_path)
    test_coords = np.load(test_idx_path)

    print(f"Total test pixels available: {len(test_coords)}")
    if args.max_samples and len(test_coords) > args.max_samples:
        print(f"Subsampling to {args.max_samples} pixels for t-SNE.")
        idx = np.random.choice(len(test_coords), args.max_samples, replace=False)
        test_coords = test_coords[idx]

    # --- FIX: Pass the PCA-transformed data directly to the dataset ---
    train_ds_for_stats = HyperspectralDataset(
        cube_path=model_args.cube_path, gt_path=model_args.gt_path,
        patch_size=model_args.patch_size, mode="train", indices=train_coords, cfg=model_args,
        preloaded_cube=img_arr  # Pass the PCA-reduced cube here
    )
    
    test_ds = HyperspectralDataset(
        cube_path=model_args.cube_path, gt_path=model_args.gt_path,
        patch_size=model_args.patch_size, mode="test", indices=test_coords,
        mean=train_ds_for_stats.mean, std=train_ds_for_stats.std, augment=False, cfg=model_args,
        preloaded_cube=img_arr  # And also here
    )
    
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    n_classes = len(np.unique(gt_arr)) - 1
    # CRITICAL FIX: Use the correctly determined number of input bands to build the model
    model = _build_model(model_args, in_ch=model_in_bands, n_cls=n_classes).to(device)
    
    try:
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
        print("✅ Successfully loaded model weights with strict=True.")
    except RuntimeError as e:
        print(f"⚠️ Strict loading failed: {e}. Attempting non-strict load.")
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)

    feats, y = extract_features(model, loader, device)

    if getattr(args, "standardize_features", True):
        feats = StandardScaler().fit_transform(feats)
    
    print("Running t-SNE...")
    tsne = TSNE(
        n_components=2,
        random_state=args.seed,
        perplexity=getattr(args, "perplexity", 50),
        n_iter=getattr(args, "n_iter", 2000),
        learning_rate=getattr(args, "learning_rate", "auto"),
        early_exaggeration=getattr(args, "early_exaggeration", 12.0),
        init=getattr(args, "init", "pca"),
        metric=getattr(args, "metric", "euclidean"),
    )
    feats_2d = tsne.fit_transform(feats)

    labels_full = get_dataset_labels(model_args.dataset)
    n_classes_plot = len(labels_full)
    cmap, norm = get_discrete_hsi_cmap(n_classes_plot)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        feats_2d[:, 0], feats_2d[:, 1], c=y + 1, cmap=cmap, norm=norm, s=10, alpha=0.8
    )
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.title(f"t-SNE Feature Visualization for {DATASET_NAME_MAP[model_args.dataset]}", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    unique_labels = np.unique(y)
    handles = [mpatches.Patch(color=cmap(cls + 1), label=labels_full[cls + 1]) for cls in unique_labels if cls >= 0]
    plt.legend(
        handles=handles, title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.
    )
    
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ t-SNE plot saved to {out_path}")

    return str(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate t-SNE plot from a trained model checkpoint.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., Pavia_University).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.pth file).")
    parser.add_argument("--output_dir", type=str, default=str(FIGURE_PATH), help="Directory to save the plot.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for feature extraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for t-SNE and subsampling.")
    parser.add_argument("--max_samples", type=int, default=5000, help="Maximum number of samples for the plot.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing plot file.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")  # <-- Add this line
    parser.add_argument("--perplexity", type=float, default=50, help="t-SNE perplexity.")
    parser.add_argument("--n_iter", type=int, default=2000, help="t-SNE iterations.")
    parser.add_argument("--learning_rate", type=str, default="auto", help="t-SNE learning rate.")
    parser.add_argument("--early_exaggeration", type=float, default=12.0, help="t-SNE early exaggeration.")
    parser.add_argument("--init", type=str, default="pca", help="t-SNE init method.")
    parser.add_argument("--metric", type=str, default="euclidean", help="t-SNE distance metric.")
    parser.add_argument("--standardize_features", action="store_true", help="Standardize features before t-SNE.")
    args = parser.parse_args()
    tsne_main(args)
    
    # example usage:
    # python src/training/tsne_plot.py --dataset Houston13 --checkpoint models/final/smbft/Houston13/smbft_best_Houston13.pth --force