# src/training/generate_optuna_reports.py

#!/usr/bin/env python3
"""
Visualize and report on Optuna studies (.pkl or DB).
Usage:
# For .pkl file (provide the full path):
python src/training/generate_optuna_reports.py --pkl_path reports/results/smbft_tuning_20251023_204712/study_Houston13_smbft_coarse.pkl --out_dir reports/optuna_reports/Houston13_coarse

# For DB studies (the script constructs the path and study name):
python src/training/generate_optuna_reports.py --dataset Houston13 --db_dir reports/optuna_db --out_dir reports/optuna_reports/Houston13 --top_n 10
python src/training/generate_optuna_reports.py --dataset Pavia_University --db_dir reports/optuna_db --out_dir reports/optuna_reports/Pavia_University --top_n 10
"""

import argparse
import os
import json
import zipfile
from collections import Counter
from pathlib import Path

import optuna
import pandas as pd
import optuna.visualization as vis
import joblib

def visualize_study(study, out_dir, phase=""):
    """Generates and saves standard Optuna visualization plots."""
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{phase}" if phase else ""
    
    # Plot optimization history
    fig1 = vis.plot_optimization_history(study)
    fig1.write_image(str(Path(out_dir) / f"optimization_history{suffix}.png"))

    # Plot parameter importances, with error handling
    try:
        fig2 = vis.plot_param_importances(study)
        fig2.write_image(str(Path(out_dir) / f"param_importances{suffix}.png"))
    except (ValueError, ZeroDivisionError):
        print(f"  - Skipping parameter importances plot for '{phase}': Not enough completed trials or variation.")
        
    # Plot parallel coordinate and slice plots
    fig3 = vis.plot_parallel_coordinate(study)
    fig3.write_image(str(Path(out_dir) / f"parallel_coordinate{suffix}.png"))
    fig4 = vis.plot_slice(study)
    fig4.write_image(str(Path(out_dir) / f"slice_plot{suffix}.png"))
    
    print(f"  - Visualizations for '{phase}' saved to {out_dir}")

def process_study(dataset: str, phase: str, model_type: str, db_dir: str, out_dir: str, top_n: int):
    """Loads a study from a DB, generates reports, and visualizes it."""
    # Construct the study name and DB path to match the tuning script
    study_name = f"{dataset}_{model_type}_{phase}"
    db_path = Path(db_dir) / f"study_{study_name}.db"
    
    if not db_path.exists():
        print(f"  - DB file not found for phase '{phase}': {db_path}. Skipping.")
        return

    storage = f"sqlite:///{db_path}"
    phase_dir = Path(out_dir) / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing phase: '{phase}' for dataset: '{dataset}'")
    print(f"  - Loading study '{study_name}' from '{db_path}'")
    
    study = optuna.load_study(study_name=study_name, storage=storage)
    visualize_study(study, phase_dir, phase)
    
    # Export full trials table
    df = study.trials_dataframe(attrs=("number", "value", "state", "params"))
    csv_all = phase_dir / f"{phase}_all_trials.csv"
    df.to_csv(csv_all, index=False)
    
    # Export Top-N trials if any have completed
    completed_df = df[df['state'] == 'COMPLETE']
    if not completed_df.empty:
        top_df = completed_df.sort_values("value", ascending=False).head(top_n)
        csv_top = phase_dir / f"{phase}_top_{top_n}_trials.csv"
        top_df.to_csv(csv_top, index=False)
        
        # Calculate robustness stats on top-N
        vals = top_df["value"]
        stats = {"mean_top_n": float(vals.mean()), "std_top_n": float(vals.std())}
        json_stats = phase_dir / f"{phase}_robustness_stats.json"
        with open(json_stats, "w") as f:
            json.dump(stats, f, indent=2)
    
    # Save trial-state summary
    states = Counter(df["state"])
    json_states = phase_dir / f"{phase}_state_summary.json"
    with open(json_states, "w") as f:
        json.dump(dict(states), f, indent=2)
        
    print(f"  - Reports for '{phase}' written to {phase_dir}")

def package_reports(out_dir: str, zip_name: str):
    """Packages all generated report files into a single ZIP archive."""
    zip_path = Path(out_dir) / zip_name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(out_dir):
            for fn in files:
                if fn == zip_name:
                    continue
                full_path = Path(root) / fn
                arc_name = full_path.relative_to(out_dir)
                zf.write(full_path, arcname=arc_name)
    print(f"\n❏ Packaged all reports into {zip_path}")

def main():
    p = argparse.ArgumentParser(description="Visualize and report on Optuna studies (.pkl or DB).")
    p.add_argument("--pkl_path", type=str, help="Full path to Optuna study .pkl file")
    p.add_argument("--dataset", type=str, help="Dataset name (for DB mode)")
    p.add_argument("--model_type", default="smbft", help="Model type prefix used in study names")
    p.add_argument("--phases", nargs="+", default=["coarse", "fine"], help="Optuna study phases to process (DB mode)")
    p.add_argument("--db_dir", default="reports/optuna_db", help="Directory containing study DBs")
    p.add_argument("--out_dir", default="reports/optuna_reports", help="Where to write all reports/figures")
    p.add_argument("--top_n", type=int, default=10, help="How many top trials to include in reports (DB mode)")
    p.add_argument("--zip_name", help="Name of the output ZIP file (optional)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.pkl_path:
        print(f"Loading Optuna study from: {args.pkl_path}")
        study = joblib.load(args.pkl_path)
        visualize_study(study, args.out_dir)
        print("\n✅ Visualizations for .pkl study complete.")
    elif args.dataset:
        for phase in args.phases:
            process_study(args.dataset, phase, args.model_type, args.db_dir, args.out_dir, args.top_n)
        
        zip_name = args.zip_name or f"optuna_reports_{args.dataset}.zip"
        package_reports(args.out_dir, zip_name)
        print("\n✅ All DB-based reports and visualizations complete.")
    else:
        print("❌ Please provide either --pkl_path or --dataset.")
        return 
    print(f"\n✅ All reports saved to {args.out_dir}")

if __name__ == "__main__":
    main()
