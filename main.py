"""SMBFT entry-point CLI.

Runs training or testing for the selected dataset with reproducible
splits, optional PCA, and optional t-SNE visualization.
"""

# main.py

import os
import argparse
import sys
from pathlib import Path

# Dynamically add the project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from src.training.train import train
from src.data.data_loader import load_dataset, DATASET_PATHS
from src.utils.utils import set_seed, apply_pca
from src.training.test import test
from src.data.dataset_info import get_imbalance_ratio
from src.training.tsne_plot import tsne_main
from src.utils.utils import apply_pca
# -----------------------------------------------------------------------------#
def resolve_dataset_paths(dataset_name: str) -> tuple[Path, Path]:
    """Return absolute (cube_path, gt_path) from DATASET_PATHS entry."""
    cfg = DATASET_PATHS[dataset_name]
    # The paths in DATASET_PATHS are already absolute, no need to prepend PROJECT_ROOT
    return (cfg["image"], cfg["ground_truth"])

def main() -> None:
    p = argparse.ArgumentParser("SMBFT Hyperspectral Image Classification")

    # --- Execution Mode ---
    p.add_argument("--mode", choices=["train", "test"], default="train", help="Set to 'train' for training or 'test' for evaluation.")
    p.add_argument("--checkpoint", type=str, help="Path to checkpoint file. Required for --mode test.")

    # --- Data & Splitting ---
    p.add_argument("--dataset", type=str, choices=list(DATASET_PATHS.keys()), default="Houston13", help="Name of the dataset to use.")
    p.add_argument("--train_ratio", type=float, default=0.05, help="Fraction of samples for the training set.")
    p.add_argument("--val_ratio", type=float, default=0.05, help="Fraction of samples for the validation set.")
    p.add_argument("--patch_size", type=int, default=11, help="Side length of the input square patches.")
    p.add_argument("--stride", type=int, default=5, help="Stride for patch extraction during training.")
    p.add_argument("--pca_components", type=int, default=0, help="Number of PCA components to use. Set to 0 to disable.")

    # --- Optimization & Training Loop ---
    p.add_argument("--epochs", type=int, default=200, help="Total number of training epochs.")
    p.add_argument("--early_stop", type=int, default=20, help="Patience for early stopping based on validation accuracy.")
    p.add_argument("--batch_size", type=int, default=128, help="Number of samples per batch.")
    p.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate for the AdamW optimizer.")
    p.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for the AdamW optimizer.")
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    p.add_argument("--recon_weight", type=float, default=1.0, help="Weight for the spectral reconstruction loss.")

    # --- Model Hyperparameters ---
    p.add_argument("--spec_dim", type=int, default=64, help="Embedding dimension for the spectral stream.")
    p.add_argument("--spat_dim", type=int, default=64, help="Embedding dimension for the spatial stream.")
    p.add_argument("--n_heads", type=int, default=8, help="Number of attention heads in the Transformer.")
    p.add_argument("--n_layers", type=int, default=4, help="Number of layers in each Transformer encoder.")
    p.add_argument("--fusion", choices=["graph", "query"], default="graph", help="Final fusion strategy ('graph' or 'query' attention).")
    p.add_argument("--mask_ratio", type=float, default=0.3, help="Ratio of spectral bands to mask for the reconstruction task.")
    p.add_argument("--mask_mode", type=str, choices=["random", "block"], default="random", help="Masking mode ('random' or 'block').")
    p.add_argument("--dim_feedforward", type=int, default=512, help="Dimension of the feedforward network in the Transformer.")
    p.add_argument("--d3_out_channels", type=int, default=16, help="Number of output channels for the initial 3D CNN block.")


    # --- Augmentation & Ablation Flags ---
    p.add_argument("--no_band_dropout", action="store_true", help="Disable band dropout augmentation.")
    p.add_argument("--no_spec_jitter", action="store_true", help="Disable spectral jitter augmentation.")
    p.add_argument("--no_spec_shuffle", action="store_true", help="Disable spectral shuffle augmentation.")
    p.add_argument("--no_mixup", action="store_true", help="Disable mixup augmentation during training.")
    p.add_argument("--no_cutmix", action="store_true", help="Disable cutmix augmentation during training.")
    p.add_argument("--no_spec_mask", action="store_true", help="Disable the entire spectral masking and reconstruction task.")
    p.add_argument("--no_cross_attn", action="store_true", help="Disable cross-attention in bidirectional transformer")

    # --- System & Reproducibility ---
    p.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for the DataLoader.")
    p.add_argument("--output_dir", type=str, default="./models/checkpoints", help="Directory to save checkpoints and training logs.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # --- Post-hoc Analysis ---
    p.add_argument("--run_tsne", action="store_true", help="Run t-SNE visualization after training or testing.")
    p.add_argument("--max_samples", type=int, default=3000, help="Max number of samples for t-SNE.")
    p.add_argument("--force", action="store_true", help="Force overwrite of existing t-SNE plot if it exists.")

    args = p.parse_args()

    # ── Reproducibility ----------------------------------------------------
    set_seed(args.seed)

    # ── Load data arrays & resolve file paths ------------------------------
    img_arr, gt_arr = load_dataset(args.dataset)
    # IMPORTANT: Do NOT apply PCA here. PCA is fit on the train split in train.py
    # and the state (mean, components) is reused at test/tsne time.
    args.raw_gt = gt_arr
    args.raw_cube = img_arr
    args.imbalance_ratio = get_imbalance_ratio(gt_arr)
    args.cube_path, args.gt_path = resolve_dataset_paths(args.dataset)

    # ── Create output directory --------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    # ----------------------------------------------- Dispatch -------------
    if args.mode == "train":
        train(args)
        if args.run_tsne:
            print("\n--- Running t-SNE analysis after training ---")
            if not args.checkpoint:
                args.checkpoint = os.path.join(
                    args.output_dir, f"smbft_best_{args.dataset}.pth"
                )
            # The tsne_plot script will now load args from the checkpoint
            # So we just need to call it with the basic arguments.
            # A more robust way would be to pass the args object directly,
            # but for simplicity, we rely on the checkpoint.
            tsne_args = ["--dataset", args.dataset, "--checkpoint", args.checkpoint]
            # A placeholder for a more direct call if you refactor tsne_main
            tsne_main(args)
    else:  # test
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for --mode test")
        test(args)
        if args.run_tsne:
            print("\n--- Running t-SNE analysis after testing ---")
            tsne_main(args)


if __name__ == "__main__":
    main()
# usage: For running you can use cli given in reports/all_dataset_clis.txt
