"""Compute and save class distributions for dataset splits.

Generates per-class counts for train/val/test splits and writes a
summary report to reports/results.
"""

# src/data/class_distribution.py

import argparse
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import load_dataset, DATASET_PATHS
from src.data.dataset_info import get_dataset_labels, get_imbalance_ratio  
from src.utils.utils import stratified_min_samples_split

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

OUTPUT_PATH = PROJECT_ROOT / "reports" / "results"


def class_distribution(gt, indices):
    labels, counts = np.unique(gt[indices[:, 0], indices[:, 1]], return_counts=True)
    return dict(zip(labels, counts))


def get_table(dataset, train_ratio, val_ratio, seed, min_samples):
    _, gt = load_dataset(dataset)
    all_idx = np.argwhere(gt > 0)
    labels  = gt[all_idx[:, 0], all_idx[:, 1]]

    # Compute floors based on class counts for a more customized split
    coords = np.argwhere(gt > 0)
    labels = gt[coords[:, 0], coords[:, 1]] - 1

    class_counts = np.bincount(labels)
    max_allowed_train_floor = int(min(class_counts) * 0.8)
    train_floor = min(10, max_allowed_train_floor)
    max_allowed_val_floor = int(min(class_counts) * 0.8)
    val_floor = min(2, max_allowed_val_floor)

    # Use stratified splitter (only one call is now used)
    train_idx, val_idx, test_idx = stratified_min_samples_split(
        coords,
        labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        train_min_frac=train_ratio,  # minimum fraction for training
        val_min_frac=0.0,            # no minimum fraction for validation
        train_floor=train_floor,      # minimum samples for training
        val_floor=val_floor,          # minimum samples for validation
        seed=seed
    )

    train_dist = class_distribution(gt, train_idx)
    val_dist   = class_distribution(gt, val_idx)
    test_dist  = class_distribution(gt, test_idx)

    class_names = get_dataset_labels(dataset)
    all_classes = sorted(set(train_dist) | set(val_dist) | set(test_dist))
    all_classes = [cls for cls in all_classes if cls > 0]

    table = []
    for cls in all_classes:
        name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
        table.append([
            int(cls)-1,
            name,
            train_dist.get(cls, 0),
            val_dist.get(cls, 0),
            test_dist.get(cls, 0)
        ])
    headers = ["Class", "Name", "Train", "Val", "Test"]

    # Calculate imbalance ratio for train split
    train_labels = gt[train_idx[:, 0], train_idx[:, 1]]
    imbalance_ratio = get_imbalance_ratio(train_labels)
    loss_metric = "FocalLoss" if imbalance_ratio > 5 else "CrossEntropy"
    return headers, table, imbalance_ratio, loss_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ratio", type=float, default=0.05)
    parser.add_argument("--val_ratio",   type=float, default=0.05)
    parser.add_argument("--min_samples", type=int,   default=5,
                        help="Minimum samples per class in train & val")
    parser.add_argument("--seed",        type=int,   default=242)
    parser.add_argument("--output",      type=str,   default="class_distributions.txt",
                        help="Output file name under reports/results/")
    args = parser.parse_args()

    dataset_names = list(DATASET_PATHS.keys())
    output_lines = []

    for dataset in dataset_names:
        headers, table, imbalance_ratio, loss_metric = get_table(
            dataset,
            args.train_ratio,
            args.val_ratio,
            args.seed,
            args.min_samples
        )
        title = (
            f"🛢️ Dataset: {dataset}\n"
            f"✂️ Train ratio: {args.train_ratio}  Val ratio: {args.val_ratio}  Min samples: {args.min_samples}\n"
            f"⚖️ Imbalance ratio: {imbalance_ratio:.2f}   Loss: {loss_metric}\n"
        )
        if tabulate:
            table_str = tabulate(table, headers=headers, tablefmt="github")
        else:
            table_str = "\t".join(headers) + "\n" + "\n".join(
                "\t".join(str(x) for x in row) for row in table
            )
        output_lines.append(title)
        output_lines.append(table_str)
        output_lines.append("")

    output_txt = "\n".join(output_lines)
    print(output_txt)

    # Save to file
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_PATH / args.output
    with open(out_file, "w") as f:
        f.write(output_txt)
    print(f"✅ Class distributions saved to {out_file}")
