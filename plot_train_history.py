#!/usr/bin/env python3

"""plot_train_history.py

Plot training curves from SMBFT `train_history_*.json` files.

The training routine writes JSON files that contain:
- `epoch_metrics`: per-epoch values (train_loss, train_acc, val_oa, lr)
- `best_val_OA`, `test_OA`, `train_time_minutes`, `num_parameters`, etc.
- `hyperparameters`: run configuration (seed, dataset, PCA settings, SSL flags, ...)

This script turns one or many such JSON files into standard plots:
- Train loss vs epoch
- Train accuracy vs epoch
- Validation OA vs epoch
- Learning rate vs epoch

It can also create an overlay of multiple runs and optionally plot the
mean ± std envelope across runs.

Examples:
    # Plot a single run
    python plot_train_history.py --inputs models/final/smbft/Houston13/train_history_Houston13_seed42.json

    # Plot all seeds for a dataset folder
    python plot_train_history.py --glob 'models/final/smbft/Houston13/train_history_*.json'

    # Plot ablation-study histories
    python plot_train_history.py --glob 'reports/ablation_studies/Houston13/**/train_history_*.json' --avg

Outputs:
    By default saves figures and a summary CSV under `reports/figures/train_history/`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HistoryRun:
    path: Path
    dataset: str
    seed: Optional[int]
    hyperparameters: Dict[str, Any]
    epoch_df: pd.DataFrame
    best_val_oa: Optional[float]
    test_oa: Optional[float]
    train_time_minutes: Optional[float]
    num_parameters: Optional[int]


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def load_train_history(path: Path) -> HistoryRun:
    with open(path, "r") as f:
        data = json.load(f)

    dataset = str(data.get("dataset", "unknown"))
    hparams = data.get("hyperparameters", {}) or {}
    seed = _safe_int(hparams.get("seed"))

    epoch_metrics = data.get("epoch_metrics") or []
    if not isinstance(epoch_metrics, list) or len(epoch_metrics) == 0:
        raise ValueError(f"No 'epoch_metrics' found in {path}")

    df = pd.DataFrame(epoch_metrics)
    # Normalize expected column names (old/new variants).
    if "val_OA" in df.columns and "val_oa" not in df.columns:
        df = df.rename(columns={"val_OA": "val_oa"})

    # Ensure required columns exist if possible.
    for col in ["epoch", "train_loss", "train_acc", "val_oa", "lr"]:
        if col not in df.columns:
            df[col] = np.nan

    df = df.sort_values("epoch").reset_index(drop=True)

    return HistoryRun(
        path=path,
        dataset=dataset,
        seed=seed,
        hyperparameters=hparams,
        epoch_df=df,
        best_val_oa=_safe_float(data.get("best_val_OA")),
        test_oa=_safe_float(data.get("test_OA")),
        train_time_minutes=_safe_float(data.get("train_time_minutes")),
        num_parameters=_safe_int(data.get("num_parameters")),
    )


def _label_for_run(run: HistoryRun) -> str:
    parts = [run.dataset]
    if run.seed is not None:
        parts.append(f"seed{run.seed}")

    # Light context that helps compare runs.
    fusion = run.hyperparameters.get("fusion")
    if fusion:
        parts.append(f"fusion={fusion}")

    pca = run.hyperparameters.get("pca_components")
    if isinstance(pca, (int, float)) and int(pca) > 0:
        parts.append(f"pca={int(pca)}")

    no_ssl = run.hyperparameters.get("no_spec_mask")
    if isinstance(no_ssl, bool):
        parts.append("ssl=off" if no_ssl else "ssl=on")

    return ", ".join(parts)


def _align_metric(runs: Sequence[HistoryRun], metric: str) -> pd.DataFrame:
    """Return a wide dataframe indexed by epoch with one column per run."""
    series_list = []
    for run in runs:
        df = run.epoch_df[["epoch", metric]].dropna()
        df = df.drop_duplicates("epoch")
        s = df.set_index("epoch")[metric]
        s.name = _label_for_run(run)
        series_list.append(s)

    if not series_list:
        return pd.DataFrame()

    wide = pd.concat(series_list, axis=1).sort_index()
    return wide


def _save_summary_csv(runs: Sequence[HistoryRun], out_dir: Path) -> Path:
    rows = []
    for run in runs:
        hp = run.hyperparameters
        rows.append(
            {
                "path": str(run.path),
                "dataset": run.dataset,
                "seed": run.seed,
                "best_val_OA": run.best_val_oa,
                "test_OA": run.test_oa,
                "train_time_minutes": run.train_time_minutes,
                "num_parameters": run.num_parameters,
                "fusion": hp.get("fusion"),
                "pca_components": hp.get("pca_components"),
                "no_spec_mask": hp.get("no_spec_mask"),
                "no_cross_attn": hp.get("no_cross_attn"),
                "lr": hp.get("lr"),
                "batch_size": hp.get("batch_size"),
                "patch_size": hp.get("patch_size"),
                "epochs": hp.get("epochs"),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train_history_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def plot_runs(
    runs: Sequence[HistoryRun],
    out_dir: Path,
    title: Optional[str],
    average: bool,
    dpi: int,
    show: bool,
) -> Tuple[Path, Path]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax_loss, ax_acc, ax_val, ax_lr = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    metrics = {
        "train_loss": (ax_loss, "Train Loss"),
        "train_acc": (ax_acc, "Train Accuracy"),
        "val_oa": (ax_val, "Validation OA"),
        "lr": (ax_lr, "Learning Rate"),
    }

    for metric, (ax, ylab) in metrics.items():
        wide = _align_metric(runs, metric)

        if wide.empty:
            ax.set_title(ylab)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylab)
            ax.grid(True, alpha=0.3)
            continue

        if average and wide.shape[1] > 1:
            mean = wide.mean(axis=1)
            std = wide.std(axis=1)
            ax.plot(mean.index, mean.values, label="mean", linewidth=2)
            ax.fill_between(mean.index, (mean - std).values, (mean + std).values, alpha=0.2, label="±1 std")
        else:
            for col in wide.columns:
                ax.plot(wide.index, wide[col].values, label=col, linewidth=1.5)

        ax.set_title(ylab)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)

        if metric == "lr":
            ax.set_yscale("log")

    if title:
        fig.suptitle(title)

    # Make legend manageable.
    for ax in [ax_loss, ax_acc, ax_val, ax_lr]:
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) <= 6:
            ax.legend(loc="best", fontsize=11)

    png_path = out_dir / "train_history_curves.png"
    pdf_path = out_dir / "train_history_curves.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return png_path, pdf_path


def _expand_inputs(inputs: Sequence[str], glob_pattern: Optional[str]) -> List[Path]:
    paths: List[Path] = []

    for p in inputs:
        paths.append(Path(p))

    if glob_pattern:
        # Use pathlib glob relative to cwd.
        for p in Path().glob(glob_pattern):
            paths.append(p)

    # Deduplicate + keep only JSON files that exist.
    uniq: List[Path] = []
    seen = set()
    for p in paths:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp in seen:
            continue
        seen.add(rp)
        if p.is_file() and p.suffix.lower() == ".json":
            uniq.append(p)

    return uniq


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot SMBFT train_history JSON curves (loss/acc/OA/LR).")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[],
        help="One or more train_history_*.json files.",
    )
    parser.add_argument(
        "--glob",
        default=None,
        help="Glob pattern to collect JSONs (quote it). Example: 'models/final/smbft/Houston13/train_history_*.json'",
    )
    parser.add_argument(
        "--out_dir",
        default="reports/figures/train_history",
        help="Directory to save plots and CSV summary.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title.",
    )
    parser.add_argument(
        "--avg",
        action="store_true",
        help="If multiple runs are given, plot mean ± std instead of per-run overlays.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG DPI.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively instead of just saving.",
    )

    args = parser.parse_args()

    input_paths = _expand_inputs(args.inputs, args.glob)
    if not input_paths:
        print("ERROR: No input JSON files found. Use --inputs or --glob.")
        return 2

    runs: List[HistoryRun] = []
    for p in input_paths:
        try:
            runs.append(load_train_history(p))
        except Exception as e:
            print(f"WARNING: Skipping {p} ({e})")

    if not runs:
        print("ERROR: No valid train_history JSONs could be loaded.")
        return 2

    out_dir = Path(args.out_dir)
    summary_csv = _save_summary_csv(runs, out_dir)

    # If all runs share the same dataset, write into a dataset subfolder.
    datasets = sorted({r.dataset for r in runs if r.dataset})
    plot_dir = out_dir
    if len(datasets) == 1:
        plot_dir = out_dir / datasets[0]

    png_path, pdf_path = plot_runs(
        runs=runs,
        out_dir=plot_dir,
        title=args.title,
        average=args.avg,
        dpi=args.dpi,
        show=args.show,
    )

    print(f"✅ Loaded {len(runs)} runs")
    print(f"✅ Saved summary CSV: {summary_csv}")
    print(f"✅ Saved plots: {png_path} and {pdf_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Usage for plotting all training histories:

# python plot_train_history.py --glob 'models/final/smbft/Houston13/train_history_*.json' --avg
# python plot_train_history.py --glob 'models/final/smbft/Pavia_University/train_history_*.json' --avg
# python plot_train_history.py --glob 'models/final/smbft/Salinas/train_history_*.json' --avg