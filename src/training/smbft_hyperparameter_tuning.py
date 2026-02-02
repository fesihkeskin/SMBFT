#!/usr/bin/env python3
# src/training/smbft_hyperparameter_tuning.py

"""
SMBFT-3D Hyperparameter Tuning Script
Performs a two-phase (coarse + fine) Optuna search using TPESampler and MedianPruner.
Relies on train.py to report intermediate values for pruning.
Exports fully reproducible CLI commands and Optuna visualization plots.
Correctly resumes from the last completed/running trial if interrupted.
"""

# ─────────────────────────── paths / imports ───────────────────────────
from __future__ import annotations
import os
import sys
import gc
import warnings
from datetime import datetime, timedelta
import time  # needed for time.time()
from argparse import Namespace
from pathlib import Path
import copy # For deep copying args

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.exceptions import TrialPruned
from optuna.study import Study
from optuna.trial import TrialState
import optuna.visualization as vis
import joblib # For saving study object

import torch
import numpy as np # For isnan check

# ─────────────────────────── setup ───────────────────────────
# CUDA / cuDNN settings for Reproducibility
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False # Set False for determinism

gc.collect()
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning) # Ignore TPESampler warning
# Suppress plotly/kaleido warnings if they occur during plot generation
warnings.filterwarnings("ignore", category=UserWarning, module='plotly')

# project-root on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Local imports
# NOTE: Ensure train.py has the trial.report() and trial.should_prune() calls
from src.training.train import train_with_oom_handling
from src.data.data_loader import load_dataset, DATASET_PATHS
from src.utils.utils import set_seed

# ───────────────────────────── constants ──────────────────────────────
MODEL_TYPE    = "smbft"
COARSE_TRIALS = 20
FINE_TRIALS   = 5
MAX_EPOCHS    = 300
EARLY_STOP    = 20
FIXED_SEED    = 242
NUM_WORKERS   = 4
TRAIN_RATIO   = 0.05
VAL_RATIO     = 0.05

# Unified output paths
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"{MODEL_TYPE}_tuning_{TS}"
OUT_DIR = PROJECT_ROOT / "reports" / "results" / RUN_NAME  # Timestamped results
DB_DIR  = PROJECT_ROOT / "reports" / "optuna_db"    # Permanent storage for studies

# Create directories
OUT_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

# Summary file setup
SUMMARY = OUT_DIR / f"opt_summary_{MODEL_TYPE}.txt"
with open(SUMMARY, "w") as f:
    f.write(f"SMBFT Hyperparameter Search (Coarse → Fine) - Run: {RUN_NAME}\n" + "="*80 + "\n\n")
    f.write(f"Timestamp: {TS}\n")
    f.write(f"Settings: Coarse Trials={COARSE_TRIALS}, Fine Trials={FINE_TRIALS}, Seed={FIXED_SEED}\n")

# ───────────────────────────── save CLI helper ──────────────────────────────
def _save_cli(dataset: str, best_params: dict, phase: str):
    """
    Writes out a shell script to reproduce the best train and test commands,
    including fixed constants for full reproducibility.
    """
    base_out = PROJECT_ROOT / "models" / "final" / MODEL_TYPE / dataset
    checkpoint = base_out / f"smbft_best_{dataset}.pth"
    script = OUT_DIR / f"{MODEL_TYPE}_{phase}_best_{dataset}.sh"
    base_defaults = {
        "train_ratio": TRAIN_RATIO, "val_ratio": VAL_RATIO, "early_stop": EARLY_STOP,
        "epochs": best_params.get("epochs", MAX_EPOCHS), # Use exact epochs
        "num_workers": NUM_WORKERS, "seed": FIXED_SEED, "dataset": dataset,
        "recon_weight": 1.0,
    }
    full_params = {**base_defaults, **best_params}
    always_skip = {"raw_gt", "raw_cube", "cube_path", "gt_path", "imbalance_ratio", "model_arch"}
    train_only_keys = {
        "epochs", "early_stop", "recon_weight", "no_band_dropout", "no_spec_jitter",
        "no_spec_shuffle", "no_mixup", "no_cutmix", "no_spec_mask"
    }
    test_only_keys = {"run_tsne", "max_samples"}

    with open(script, "w") as out:
        out.write(f"#!/bin/bash\n\n# Best parameters from {phase} for {dataset}\n\n")
        out.write("# --- Training Command ---\n")
        train_parts = ["python main.py --mode train", f"--output_dir {base_out}"]
        for k, v in sorted(full_params.items()):
            if k in always_skip or k in test_only_keys or v is None or k == "output_dir": continue
            if isinstance(v, bool):
                if v: train_parts.append(f"--{k}")
            else: train_parts.append(f"--{k} {v}")
        out.write(" ".join(train_parts) + "\n\n")

        out.write("# --- Testing Command ---\n")
        test_parts = ["python main.py --mode test", f"--output_dir {base_out}"]
        for k, v in sorted(full_params.items()):
            if k in always_skip or k in train_only_keys or v is None or k == "output_dir": continue
            if isinstance(v, bool):
                if v: test_parts.append(f"--{k}")
            else: test_parts.append(f"--{k} {v}")
        test_parts.append(f"--checkpoint {checkpoint}")
        out.write(" ".join(test_parts) + "\n")

    os.chmod(script, 0o755)
    print(f"✅ Saved executable CLI script to: {script}")

# ───────────────────────────── objective function ──────────────────────────
def _run_optuna(
    dataset: str,
    n_trials: int, # Target number of trials
    param_space: dict[str, dict],
    study_name_suffix: str # e.g., "smbft_coarse"
) -> optuna.Study:
    """
    Sets up and runs an Optuna study using TPESampler and MedianPruner.
    Correctly resumes, saves study object and plots.
    """
    study_name = f"{dataset}_{study_name_suffix}"
    storage_path = f"sqlite:///{DB_DIR}/study_{study_name}.db"
    db_file_path = DB_DIR / f"study_{study_name}.db" # Path object for checks
    dump_path = OUT_DIR / f"study_{study_name}.pkl" # Path for joblib dump

    # --- Enhanced DB/Study Loading Info ---
    print("\n" + "="*40)
    print(f"Initializing Optuna Study: '{study_name}'")
    print(f"Storage Path: {storage_path}")
    if db_file_path.exists():
        print(f"  - Database file found.")
        try:
            db_size_kb = db_file_path.stat().st_size / 1024
            db_mtime = datetime.fromtimestamp(db_file_path.stat().st_mtime)
            print(f"  - Size: {db_size_kb:.1f} KB")
            print(f"  - Last Modified: {db_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            # Try loading to check integrity and get best value so far
            temp_study = optuna.load_study(study_name=study_name, storage=storage_path)
            print(f"  - Successfully loaded. Contains {len(temp_study.trials)} trials.")
            try:
                 print(f"  - Best Value (so far): {temp_study.best_value:.4f}")
            except ValueError:
                 print("  - No completed trials found yet.")
            del temp_study # Release handle
        except Exception as e:
            print(f"  - WARNING: Error accessing existing DB: {e}")
    else:
        print(f"  - Database file does not exist. Will be created.")
    print("="*40 + "\n")
    # --- End Enhanced Info ---

    # Load data once
    cube, raw_gt = load_dataset(dataset)
    gt_path = DATASET_PATHS[dataset]["ground_truth"]
    current_run_best_val_oa = -1.0
    current_run_best_params = {}

    def objective(trial: optuna.Trial) -> float:
        nonlocal current_run_best_val_oa, current_run_best_params
        set_seed(FIXED_SEED)
        torch.cuda.empty_cache()
        sampled = {}
        for name, spec in param_space.items():
            suggest_method = getattr(trial, f"suggest_{spec['type']}")
            kwargs = {k: v for k, v in spec.items() if k != 'type'}
            sampled[name] = suggest_method(name, **kwargs)

        args = Namespace(
            dataset=dataset, cube_path=str(DATASET_PATHS[dataset]["image"]),
            gt_path=str(gt_path),
            raw_cube=cube,  # shallow copy for speed/memory
            raw_gt=raw_gt,  # shallow copy for speed/memory
            train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
            patch_size=int(sampled.get("patch_size")), stride=int(sampled.get("stride", 1)),
            mode="train", epochs=int(sampled.get("epochs", MAX_EPOCHS)),
            early_stop=EARLY_STOP, batch_size=int(sampled.get("batch_size")),
            lr=sampled.get("lr"), weight_decay=sampled.get("weight_decay"),
            dropout=sampled.get("dropout", 0.0), recon_weight=1.0,
            seed=FIXED_SEED, num_workers=NUM_WORKERS,
            output_dir=OUT_DIR / dataset / study_name_suffix / f"trial_{trial.number}",
            model_arch="smbft", spec_dim=int(sampled.get("spec_dim")),
            spat_dim=int(sampled.get("spat_dim")), n_heads=int(sampled.get("n_heads")),
            n_layers=int(sampled.get("n_layers")), fusion=sampled.get("fusion"),
            mask_ratio=sampled.get("mask_ratio"), mask_mode=sampled.get("mask_mode"),
            dim_feedforward=int(sampled.get("dim_feedforward")),
            d3_out_channels=int(sampled.get("d3_out_channels")),
            no_band_dropout=sampled.get("no_band_dropout", False),
            no_spec_jitter=sampled.get("no_spec_jitter", False),
            no_spec_shuffle=sampled.get("no_spec_shuffle", False),
            no_mixup=sampled.get("no_mixup", False),
            no_cutmix=sampled.get("no_cutmix", False),
            no_spec_mask=sampled.get("no_spec_mask", False),
            no_cross_attn=sampled.get("no_cross_attn", False),  
            pca_components=int(sampled.get("pca_components", 0)),
        )
        os.makedirs(args.output_dir, exist_ok=True)

        if args.pca_components > 0:
            print(f"   Applying PCA ({args.pca_components} components) for trial {trial.number}...")
            from src.utils.utils import apply_pca
            try: args.raw_cube = apply_pca(args.raw_cube, num_components=args.pca_components)
            except Exception as pca_e:
                print(f"   ERROR applying PCA: {pca_e}. Pruning.")
                raise TrialPruned("PCA application failed")

        try:
            val_oa = train_with_oom_handling(args, trial)
            if np.isnan(val_oa):
                print(f"Warning: Trial {trial.number} returned NaN. Pruning.")
                raise TrialPruned("NaN validation OA returned")
        except TrialPruned: raise
        except Exception as e:
            print(f"ERROR: Trial {trial.number} failed unexpectedly: {e}")
            raise TrialPruned(f"Unexpected error: {e}")

        if val_oa > current_run_best_val_oa:
             current_run_best_val_oa = val_oa
             current_run_best_params = trial.params
        return val_oa

    # --- Study Creation and Optimization ---
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
    sampler = TPESampler(seed=FIXED_SEED, n_startup_trials=10, multivariate=True, constant_liar=True)

    study = optuna.create_study(
        direction="maximize", pruner=pruner, sampler=sampler,
        study_name=study_name, storage=storage_path, load_if_exists=True
    )

    # Robust Calculation of Trials to Run
    finished_states = [TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL]
    inflight_states = [TrialState.RUNNING, TrialState.WAITING]
    finished_trials = study.get_trials(deepcopy=False, states=finished_states)
    inflight_trials = study.get_trials(deepcopy=False, states=inflight_states)
    n_finished = len(finished_trials)
    n_inflight = len(inflight_trials)
    trials_to_run = max(0, n_trials - n_finished - n_inflight)

    print(f"\nStudy '{study_name}':")
    print(f"  - Target Trials: {n_trials}")
    print(f"  - Finished Trials in DB: {n_finished}")
    print(f"  - In-flight Trials: {n_inflight}")

    start_time = time.time() # Timer for optimization phase
    if trials_to_run <= 0:
        if n_finished >= n_trials: print(f"✅ Target reached. Skipping optimization.")
        else: print(f"⏳ Target not yet reached, but {n_inflight} trials in progress.")
    else:
        print(f"🚀 Running {trials_to_run} more trials...")
        study.optimize(
            objective, n_trials=trials_to_run, n_jobs=1,
            gc_after_trial=True, # show_progress_bar=True
        )
        elapsed = timedelta(seconds=int(time.time() - start_time))
        print(f"Optimization phase completed in {elapsed}.")


    # --- Save Study Object and Plots ---
    try:
        joblib.dump(study, dump_path)
        print(f"💾 Saved study object to: {dump_path}")

        history_plot_path = OUT_DIR / f"{study_name}_history.png"
        importance_plot_path = OUT_DIR / f"{study_name}_importance.png"

        try:
            fig_history = vis.plot_optimization_history(study)
            fig_history.write_image(str(history_plot_path))
            print(f"📊 Saved optimization history plot to: {history_plot_path}")
        except Exception as e:
            print(f"⚠️ Could not generate optimization history plot: {e}")

        try:
            fig_importance = vis.plot_param_importances(study)
            fig_importance.write_image(str(importance_plot_path))
            print(f"📊 Saved parameter importance plot to: {importance_plot_path}")
        except Exception as e:
            print(f"⚠️ Could not generate parameter importance plot: {e}")
    except Exception as e:
        print(f"ERROR during study saving or plot generation: {e}")


    # Save CLI for the best trial found *within this specific script execution*
    if current_run_best_params:
         print(f"\nBest trial *within this specific run* achieved {current_run_best_val_oa:.4f} Val OA.")
         _save_cli(dataset, current_run_best_params, f"{study_name_suffix}_this_run_best")
    # Check if DB has completed trials before accessing best_value
    elif study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]):
         print(f"\nNo new best trial found in this run. Best in DB: {study.best_value:.4f}")
    else:
         print("\nNo completed trials found in the database for this study yet.")

    return study

# ───────────────────────────── main execution ────────────────────────
if __name__ == "__main__":
    from src.data.data_loader import DATASET_NAME_LIST
    DATASET_NAME_LIST = ['Houston13', 'Pavia_University', 'Salinas']

    overall_best_trials_db = {}

    for ds in DATASET_NAME_LIST:
        if ds not in DATASET_PATHS:
            print(f"❌ Dataset {ds} not found. Skipping.")
            continue
        print(f"\n{'='*30}\n🔍 Starting Tuning for Dataset: {ds}\n{'='*30}")

        with open(SUMMARY, "a") as f: f.write(f"\n\n{'='*60}\nDATASET: {ds}\n{'='*60}\n")

        # --- Phase 1: Coarse search ---
        coarse_space = {
            "epochs":         {"type":"int", "low":50, "high":MAX_EPOCHS, "step":50},
            "batch_size":     {"type":"categorical", "choices":[16, 32, 64, 128, 256]},
            "patch_size":     {"type":"categorical", "choices":[9, 11, 13, 15]},
            "stride":         {"type":"categorical", "choices":[1]},
            "lr":             {"type":"float", "low":1e-5, "high":1e-2, "log":True},
            "weight_decay":   {"type":"float", "low":1e-6, "high":1e-2, "log":True},
            "spec_dim":       {"type":"categorical", "choices":[32, 64, 128]},
            "spat_dim":       {"type":"categorical", "choices":[32, 64, 128]},
            "n_heads":        {"type":"categorical", "choices":[2, 4, 8]},
            "n_layers":       {"type":"categorical", "choices":[2, 4, 6]},
            "fusion":         {"type":"categorical", "choices":["graph", "query"]},
            "mask_ratio":     {"type":"float", "low":0.0, "high":0.5},
            "mask_mode":      {"type":"categorical", "choices":["random", "block"]},
            "dim_feedforward": {"type":"categorical", "choices":[256, 512, 1024]},
            "dropout":        {"type":"float", "low":0.0, "high":0.5},
            "d3_out_channels": {"type":"categorical", "choices":[8, 16, 32]},
            "no_band_dropout": {"type":"categorical", "choices":[True, False]},
            "no_spec_jitter":  {"type":"categorical", "choices":[True, False]},
            "no_spec_shuffle": {"type":"categorical", "choices":[True, False]},
            "no_mixup":        {"type":"categorical", "choices":[True, False]},
            "no_cutmix":       {"type":"categorical", "choices":[True, False]},
            "no_spec_mask":    {"type":"categorical", "choices":[True, False]},
            "no_cross_attn":   {"type":"categorical", "choices":[True, False]},
            "pca_components":  {"type":"categorical", "choices":[0, 15, 30, 50, 64, 100]},
        }
        coarse_study = _run_optuna(ds, COARSE_TRIALS, coarse_space, f"{MODEL_TYPE}_coarse")

        try: best_coarse_trial = coarse_study.best_trial
        except ValueError:
            print(f"❌ Coarse search for {ds} has no completed trials. Skipping fine-tuning.")
            with open(SUMMARY, "a") as f: f.write("Coarse search failed (no completed trials).\n")
            continue

        with open(SUMMARY, "a") as f:
            f.write(f"\n--- Coarse Search Best Trial (Overall in DB, Trial #{best_coarse_trial.number}) ---\n")
            f.write(f"Best Validation OA: {best_coarse_trial.value:.4f}\n")
            f.write("Best Hyperparameters (Coarse):\n")
            for key, value in sorted(best_coarse_trial.params.items()): f.write(f"  - {key}: {value}\n")

        # --- Phase 2: Fine search ---
        best_coarse_params = best_coarse_trial.params
        def _get_neighbors(val, choices=[32, 48, 64, 96, 128]):
             choices = sorted(list(set(choices)))
             try:
                 idx = choices.index(val)
                 low_idx, high_idx = max(0, idx - 1), min(len(choices)-1, idx + 1)
                 return sorted(list(set([choices[low_idx], val, choices[high_idx]])))
             except ValueError: return [val]

        fine_space = {
            "lr":           {"type": "float", "low": best_coarse_params["lr"] * 0.5, "high": best_coarse_params["lr"] * 2, "log": True},
            "weight_decay": {"type": "float", "low": max(1e-9, best_coarse_params["weight_decay"] * 0.5), "high": min(1e-1, best_coarse_params["weight_decay"] * 2), "log": True},
            "mask_ratio":   {"type": "float", "low": max(0.0, best_coarse_params["mask_ratio"] - 0.1), "high": min(0.7, best_coarse_params["mask_ratio"] + 0.1)},
            "epochs":       {"type": "int", "low": max(50, best_coarse_params["epochs"] - 50), "high": min(MAX_EPOCHS + 50, best_coarse_params["epochs"] + 50), "step": 10},
            "n_layers":     {"type": "int", "low": max(1, best_coarse_params["n_layers"] - 1), "high": best_coarse_params["n_layers"] + 1},
            "dropout":      {"type": "float", "low": max(0.0, best_coarse_params["dropout"] - 0.1), "high": min(0.7, best_coarse_params["dropout"] + 0.1)},
            "spec_dim":     {"type":"categorical", "choices": _get_neighbors(best_coarse_params["spec_dim"])},
            "spat_dim":     {"type":"categorical", "choices": _get_neighbors(best_coarse_params["spat_dim"])},
            "batch_size":   {"type":"categorical", "choices":[best_coarse_params["batch_size"]]},
            "patch_size":   {"type":"categorical", "choices":[best_coarse_params["patch_size"]]},
            "stride":       {"type":"categorical", "choices":[1]},
            "n_heads":      {"type":"categorical", "choices":[best_coarse_params["n_heads"]]},
            "fusion":       {"type":"categorical", "choices":[best_coarse_params["fusion"]]},
            "mask_mode":    {"type":"categorical", "choices":[best_coarse_params["mask_mode"]]},
            "dim_feedforward": {"type":"categorical", "choices":[best_coarse_params["dim_feedforward"]]},
            "d3_out_channels": {"type":"categorical", "choices":[best_coarse_params["d3_out_channels"]]},
            "no_band_dropout": {"type":"categorical", "choices":[best_coarse_params["no_band_dropout"]]},
            "no_spec_jitter":  {"type":"categorical", "choices":[best_coarse_params["no_spec_jitter"]]},
            "no_spec_shuffle": {"type":"categorical", "choices":[best_coarse_params["no_spec_shuffle"]]},
            "no_mixup":        {"type":"categorical", "choices":[best_coarse_params["no_mixup"]]},
            "no_cutmix":       {"type":"categorical", "choices":[best_coarse_params["no_cutmix"]]},
            "no_spec_mask":    {"type":"categorical", "choices":[best_coarse_params["no_spec_mask"]]},
            "no_cross_attn":   {"type":"categorical", "choices":[best_coarse_params["no_cross_attn"]]},
            "pca_components":  {"type":"categorical", "choices":[best_coarse_params["pca_components"]]},
        }
        fine_study = _run_optuna(ds, FINE_TRIALS, fine_space, f"{MODEL_TYPE}_fine")

        try:
             best_fine_trial = fine_study.best_trial
             if best_fine_trial.value >= best_coarse_trial.value:
                 overall_best_trials_db[ds] = (best_fine_trial, "fine")
             else:
                 overall_best_trials_db[ds] = (best_coarse_trial, "coarse")
        except ValueError:
             print(f"Fine search for {ds} has no completed trials in DB. Using coarse best.")
             overall_best_trials_db[ds] = (best_coarse_trial, "coarse")

        with open(SUMMARY, "a") as f:
            try:
                 best_fine_trial_for_log = fine_study.best_trial
                 f.write(f"\n--- Fine Search Best Trial (Overall in DB, Trial #{best_fine_trial_for_log.number}) ---\n")
                 f.write(f"Best Validation OA: {best_fine_trial_for_log.value:.4f}\n")
                 f.write("Best Hyperparameters (Fine):\n")
                 for key, value in sorted(best_fine_trial_for_log.params.items()): f.write(f"  - {key}: {value}\n")
            except ValueError:
                 f.write("\n--- Fine Search has no completed trials ---\n")

            best_trial_obj_db, best_phase_db = overall_best_trials_db[ds]
            f.write(f"\n--- OVERALL BEST (from DB, {best_phase_db} phase trial #{best_trial_obj_db.number}) ---\n")
            f.write(f"Best Validation OA: {best_trial_obj_db.value:.4f}\n")
            _save_cli(ds, best_trial_obj_db.params, f"overall_best_from_db_{best_phase_db}")

    print(f"\n{'='*80}")
    print("🎉 All Tuning Completed!")
    print(f"   Summary saved to: {SUMMARY}")
    print(f"   Detailed results, scripts, and logs are in: {OUT_DIR}")
    print(f"   Optuna study databases are in: {DB_DIR}")
    print(f"{'='*80}")

    if COARSE_TRIALS > 10:
        print("\nTriggering system shutdown (commented out)...")
        os.system("sudo /sbin/shutdown -h +1")  # Shutdown after 1 minute
        pass
