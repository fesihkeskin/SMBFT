#!/usr/bin/env python3
# ablate.py

"""
Ablation study runner for the SMBFT model.

This script systematically runs training and testing experiments to evaluate the
impact of different model components and data processing choices. It requires a
set of pre-tuned, high-performing baseline hyperparameters.

The script is resumable. If interrupted, it will check for existing result files 
and skip any experiments that have already been completed.

Usage:
    python ablate.py --dataset Houston13

The script will generate an output directory structure like this:
<output_root>/
├── <dataset_name>/
│   ├── 01_baseline/
│   │   ├── seed_242/
│   │   ├── seed_343/
│   │   └── seed_454/
│   ├── 02_no_augmentations/
│   │   ├── seed_242/
│   │   └── ...
│   └── ...
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any
import subprocess

# --- SCRIPT CONFIGURATION ---
# project-root on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

print(f"INFO: Project root set to '{PROJECT_ROOT}'")
MAIN_SCRIPT_PATH = PROJECT_ROOT / "main.py"
OUTPUT_ROOT = PROJECT_ROOT / "reports" / "ablation_studies"

# Define multiple seeds for robust evaluation
SEEDS = [242, 343, 454]

# --- BASELINE HYPERPARAMETERS ---
# IMPORTANT: Replace these with the best hyperparameters found from your Optuna search.
# These will be used as the foundation for all ablation experiments.
BEST_HP = {
    "patch_size": 11,
    "batch_size": 32,
    "lr": 5e-4,
    "weight_decay": 1e-3,
    "epochs": 200,
    "early_stop": 20,
    "spec_dim": 64,
    "spat_dim": 64,
    "n_heads": 4,
    "n_layers": 4,
    "fusion": "graph",
    "mask_ratio": 0.2,
    "mask_mode": "random",
    "dim_feedforward": 512,
    "dropout": 0.2,
    "d3_out_channels": 16,
    "no_spec_mask": False,
    "no_cross_attn": False,  # baseline uses cross-attn
    "pca_components": 0, # Baseline does not use PCA unless specified
}

# --- ABLATION EXPERIMENTS DEFINITION ---
# Each key is the experiment name. The value is a dictionary of arguments
# to override from the BEST_HP baseline.
ABLATION_CONFIG = {
    # 1. The full baseline model
    "01_baseline": {},

    # 2. Data Augmentation Ablation
    "02_no_augmentations": {
        "no_band_dropout": True,
        "no_spec_jitter": True,
        "no_spec_shuffle": True,
        "no_mixup": True,
        "no_cutmix": True,
    },
    "03_no_sample_mixing": {
        "no_mixup": True,
        "no_cutmix": True,
    },
    "04_no_spectral_augmentations": {
        "no_band_dropout": True,
        "no_spec_jitter": True,
        "no_spec_shuffle": True,
    },

    # 3. Fusion Strategy Ablation
    "05_fusion_query": {
        "fusion": "query", # Query Attention Pooling
    },

    # 4. PCA Dimensionality Ablation
    "06_pca_15_components": {
        "pca_components": 15,
    },
    "07_pca_30_components": {
        "pca_components": 30,
    },
    "08_pca_50_components": {
        "pca_components": 50,
    },

    # 5. Self-Supervision (Spectral Masking) Ablation
    "09_no_spectral_masking": {
        "no_spec_mask": True,
    },

    # 6. Cross-Attention Ablation
    "10_no_cross_attn": {
        "no_cross_attn": True, 
    },
}

def run_experiment(dataset: str, exp_name: str, overrides: Dict[str, Any], seeds: List[int]):
    """
    Runs a single ablation experiment for a given dataset over multiple seeds,
    skipping any runs that have already been completed.
    """
    print(f"\n{'='*80}")
    print(f"🔬 STARTING ABLATION: '{exp_name}' on dataset '{dataset}'")
    print(f"{'='*80}")

    for seed in seeds:
        # Define output directory for this specific run
        output_dir = OUTPUT_ROOT / dataset / exp_name / f"seed_{seed}"

        # --- Check if the experiment has already been completed ---
        # The train.py script saves a history JSON file upon successful completion.
        # We use its existence to determine if we can skip this run.
        expected_json_path = output_dir / f"train_history_{dataset}_seed{seed}.json"
        if expected_json_path.exists():
            print(f"\n--- SKIPPING seed {seed} for experiment '{exp_name}' (already completed) ---")
            print(f"   Found result file: {expected_json_path}")
            continue
        # --- End of check ---

        print(f"\n--- Running seed {seed} for experiment '{exp_name}' ---")

        # Combine baseline HPs with experiment-specific overrides
        current_hp = BEST_HP.copy()
        current_hp.update(overrides)

        # Ensure the output directory exists before running
        os.makedirs(output_dir, exist_ok=True)

        # --- Construct the command for main.py ---
        # Start with the base command
        cmd = [
            "python", str(MAIN_SCRIPT_PATH),
            "--mode", "train",
            "--dataset", dataset,
            "--output_dir", str(output_dir),
            "--seed", str(seed),
        ]

        # Add hyperparameters to the command
        for key, value in current_hp.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.append(f"--{key}")
                cmd.append(str(value))

        # --- Print and run the command ---
        # Use shell=False for better security and argument handling
        print("Running command:")
        print(" ".join(cmd))

        try:
            subprocess.run(cmd, check=True)
            print(f"✅ SUCCESS: Seed {seed} for '{exp_name}' completed.")
            print(f"   Results saved in: {output_dir}")

        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR: Seed {seed} for '{exp_name}' failed with exit code {e.returncode}.")
            print("   Check the output above for error messages.")
        except KeyboardInterrupt:
            print("\n🛑 User interrupted. Aborting script.")
            exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation studies for the SMBFT model.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to run ablations on (e.g., Houston13)."
    )
    args = parser.parse_args()

    # Check if main.py exists
    if not MAIN_SCRIPT_PATH.is_file():
        print(f"FATAL: main.py not found at '{MAIN_SCRIPT_PATH}'")
        exit(1)

    # Run all defined ablation experiments
    for name, overrides in ABLATION_CONFIG.items():
        run_experiment(args.dataset, name, overrides, SEEDS)

    print(f"\n{'='*80}")
    print("🎉 All ablation studies completed!")
    print(f"   Check the root output directory: {OUTPUT_ROOT / args.dataset}")
    print(f"{'='*80}")

# after finishing the ablation study, run shutdown computer
subprocess.run(["shutdown", "-h", "now"])
