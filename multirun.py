"""Run multiple SMBFT training/testing jobs across datasets and seeds."""

import subprocess
import os
import sys
from pathlib import Path
import json

# Dynamically add the project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project root: {PROJECT_ROOT}")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration for each dataset (matches optimum CLI exactly)
configs = [
    {
        "dataset": "Pavia_University",
        "batch_size": 128,
        "d3_out_channels": 16,
        "dim_feedforward": 512,
        "dropout": 0.004915142003102835,
        "early_stop": 30,
        "epochs": 100,
        "fusion": "query",
        "lr": 0.0023014491965656723,
        "mask_mode": "random",
        "mask_ratio": 0.24487402392504498,
        "n_heads": 4,
        "n_layers": 4,
        "no_band_dropout": True,
        "no_mixup": True,
        "no_spec_jitter": True,
        "no_spec_mask": True,
        "no_spec_shuffle": True,
        "num_workers": 4,
        "patch_size": 9,
        "pca_components": 64,
        "recon_weight": 1.0,
        "spat_dim": 32,
        "spec_dim": 64,
        "stride": 1,
        "train_ratio": 0.05,
        "val_ratio": 0.05,
        "weight_decay": 3.448231348743791e-06,
    },
    {
        "dataset": "Houston13",
        "batch_size": 16,
        "d3_out_channels": 16,
        "dim_feedforward": 512,
        "dropout": 0.008754262934426417,
        "early_stop": 30,
        "epochs": 90,
        "fusion": "query",
        "lr": 0.0009930721819500054,
        "mask_mode": "random",
        "mask_ratio": 0.22065971626008807,
        "n_heads": 4,
        "n_layers": 5,
        "no_band_dropout": True,
        "no_mixup": True,
        "no_spec_jitter": True,
        "no_spec_mask": True,
        "no_spec_shuffle": True,
        "num_workers": 4,
        "patch_size": 9,
        "pca_components": 64,
        "recon_weight": 1.0,
        "spat_dim": 128,
        "spec_dim": 64,
        "stride": 1,
        "train_ratio": 0.05,
        "val_ratio": 0.05,
        "weight_decay": 1.057882708458453e-05,
    },
    {
        "dataset": "Salinas",
        "batch_size": 16,
        "d3_out_channels": 8,
        "dim_feedforward": 512,
        "dropout": 0.3073733550267665,
        "early_stop": 30,
        "epochs": 310,
        "fusion": "query",
        "lr": 0.0001805433174407008,
        "mask_mode": "random",
        "mask_ratio": 0.13837922495315128,
        "n_heads": 2,
        "n_layers": 2,
        "no_cutmix": True,
        "no_mixup": True,
        "no_spec_mask": True,
        "num_workers": 4,
        "patch_size": 13,
        "pca_components": 30,
        "recon_weight": 1.0,
        "spat_dim": 48,
        "spec_dim": 48,
        "stride": 1,
        "train_ratio": 0.05,
        "val_ratio": 0.05,
        "weight_decay": 7.237813686384994e-06,
    },
]

# Seeds for multiple runs
seeds = list(range(242, 342, 10)) #  [242, 252, 262, 272, 282, 292, 302, 312, 322, 332]
def build_command(config, seed, out_dir, mode="train", checkpoint=None):
    """Build the command list for training or testing."""
    # Use the provided out_dir directly, which will be seed-specific
    os.makedirs(out_dir, exist_ok=True)
    
    cmd = [
        "python", "main.py",
        "--mode", mode,
        "--output_dir", out_dir,  # Use the seed-specific directory
        "--seed", str(seed),
    ]
    
    # Add all configuration parameters
    for key, value in config.items():
        if key == "dataset":
            cmd += ["--dataset", value]
        elif isinstance(value, bool):
            if value:
                cmd += [f"--{key}"]
        else:
            cmd += [f"--{key}", str(value)]
    
    if mode == "test" and checkpoint:
        cmd += ["--checkpoint", checkpoint]
    
    return cmd

for config in configs:
    dataset = config["dataset"]
    print(f"\n=== Dataset: {dataset} ===")
    base_out_dir = os.path.join(PROJECT_ROOT, "models", "final", "smbft", dataset)
    os.makedirs(base_out_dir, exist_ok=True)

    for seed in seeds:
        # Define a unique directory for this specific run
        seed_dir = os.path.join(base_out_dir, f"seed_{seed}")
        
        # Define paths for this seed
        train_history_file = os.path.join(seed_dir, f"train_history_{dataset}_seed{seed}.json")
        checkpoint_file = os.path.join(seed_dir, f"smbft_best_{dataset}.pth") # Use the default name train.py saves
        
        # Skip if the final history file already exists
        if os.path.exists(train_history_file):
            print(f" → Training history exists, skipped for seed={seed}.")
            continue
        
        # 1. Run training with the seed-specific output directory
        train_cmd = build_command(config, seed, seed_dir, mode="train")
        
        try:
            print(f"🚀 Starting training for {dataset} with seed {seed}...")
            subprocess.run(train_cmd, check=True)
            print(f"    → Training completed for seed={seed}")
        except subprocess.CalledProcessError as e:
            print(f"Training failed for {dataset}, seed={seed}. Error: {e}")
            continue

        # 2. Run test after successful training
        if os.path.exists(checkpoint_file):
            test_cmd = build_command(config, seed, seed_dir, mode="test", checkpoint=checkpoint_file)
            try:
                print(f"🚀 Starting testing for {dataset} with seed {seed}...")
                subprocess.run(test_cmd, check=True)
                print(f"    → Test completed for seed={seed}")
            except subprocess.CalledProcessError as e:
                print(f"Test failed for {dataset}, seed={seed}. Error: {e}")
        else:
            print(f"⚠️ Checkpoint not found at {checkpoint_file}, skipping test.")

print("\n✅ All runs completed.")

# Uncomment to shutdown after completion
os.system("sudo /sbin/shutdown -h now")