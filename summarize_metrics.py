"""Summarize training/testing metrics across seeds for publication."""

# summarize_metrics.py

import json
import csv
import os
import statistics
import sys
from pathlib import Path
import re

# Dynamically add the project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

def summarize_metrics(root_dir: str):
    """
    Summarize key metrics from training and testing JSON files for publication.
    Gathers OA, AA, Kappa, training time, inference time, model size, and params.
    Averages metrics across seeds, except for static params like model size.
    """
    root = Path(root_dir)
    # Filter for directories only, excluding files like .DS_Store
    datasets = sorted([p for p in root.iterdir() if p.is_dir()])
    print(f"Found {len(datasets)} datasets in {root_dir}")
    print(f"Dataset names: {[ds.name for ds in datasets]}")
    print("Summarizing metrics for publication...")
    
    summary = {}

    for ds_path in datasets:
        print(f"\nProcessing dataset: {ds_path.name}")
        
        # Metrics to be averaged across seeds
        avg_metrics = {
            "oa": [], "aa": [], "kappa": [],
            "best_val_oa": [],
            "train_time": [], 
            "infer_time_total": [],
            "infer_time_sample": [],
            "per_class_aa": {} # Will store class_name: [list_of_values]
        }
        
        # Metrics that are constant across seeds (taken from first seed)
        static_metrics = {
            "model_size": None,
            "num_params": None,
            "num_samples": None,
            "class_names": [] # To maintain order
        }
        
        seed_dirs = sorted([p for p in ds_path.glob("seed_*") if p.is_dir()])
        if not seed_dirs:
            print(f"  Skipping {ds_path.name}: No 'seed_*' directories found.")
            continue
        
        print(f"  Found {len(seed_dirs)} seeds: {[s.name for s in seed_dirs]}")

        for seed_dir in seed_dirs:
            seed_match = re.search(r'seed_(\d+)', seed_dir.name)
            if not seed_match:
                print(f"  Skipping directory {seed_dir.name}: Could not parse seed number.")
                continue
            seed = seed_match.group(1)

            # --- CORRECTED FILE PATHS ---
            train_history_file = seed_dir / f"train_history_{ds_path.name}_seed{seed}.json"
            test_metrics_file = seed_dir / "test_results" / f"metrics_seed_{seed}.json"

            if not train_history_file.exists():
                print(f"  Skipping seed {seed}: Missing train history file: {train_history_file}")
                continue
            if not test_metrics_file.exists():
                print(f"  Skipping seed {seed}: Missing test metrics file: {test_metrics_file}")
                continue

            # Extract metrics from both files
            try:
                with open(train_history_file, 'r') as f:
                    train_data = json.load(f)
                with open(test_metrics_file, 'r') as f:
                    test_data = json.load(f)

                # --- Extract Static Metrics (only once, from first valid seed) ---
                if static_metrics["model_size"] is None:
                    model_size_str = train_data.get("Model_size_MB", "0 MB")
                    # Handle potential parsing error
                    try:
                        static_metrics["model_size"] = float(model_size_str.split()[0])
                    except (ValueError, IndexError):
                         static_metrics["model_size"] = 0.0
                         print(f"  Warning: Could not parse Model_size_MB '{model_size_str}' for seed {seed}.")
                    
                    static_metrics["num_params"] = train_data.get("num_parameters", 0)
                    static_metrics["num_samples"] = test_data.get("Number of Samples", 0)

                # --- Extract Averaged Metrics ---
                # From test_metrics.json
                avg_metrics["oa"].append(test_data.get("Overall Accuracy", 0))
                avg_metrics["aa"].append(test_data.get("Average Accuracy", 0))
                avg_metrics["kappa"].append(test_data.get("Kappa", 0))
                avg_metrics["infer_time_total"].append(test_data.get("Inference Time Total (seconds)", 0))
                avg_metrics["infer_time_sample"].append(test_data.get("Inference Time Per Sample (seconds)", 0))
                
                # From train_history.json
                avg_metrics["train_time"].append(train_data.get("train_time_minutes", 0))
                avg_metrics["best_val_oa"].append(train_data.get("best_val_OA", 0))

                # Handle Per Class Accuracy
                per_class_data = test_data.get("Per Class Accuracy", {})
                if per_class_data:
                    # On the first seed, initialize the lists
                    if not avg_metrics["per_class_aa"]:
                        for class_name in per_class_data.keys():
                            avg_metrics["per_class_aa"][class_name] = []
                        static_metrics["class_names"] = list(per_class_data.keys()) # Store order
                    
                    # Append values, ensuring class name matches
                    for class_name, acc in per_class_data.items():
                        if class_name in avg_metrics["per_class_aa"]:
                            avg_metrics["per_class_aa"][class_name].append(acc)
                        elif class_name not in static_metrics["class_names"]:
                             print(f"  Warning: New class '{class_name}' found in seed {seed}, not in first seed. Skipping.")
                
                print(f"  Successfully processed seed {seed}.")

            except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                print(f"  Error processing files for seed {seed}: {e}")
                continue
        
        # Calculate mean and std dev if we have data
        if avg_metrics["oa"]:
            num_seeds = len(avg_metrics["oa"])
            print(f"  Calculating averages over {num_seeds} seed(s) for {ds_path.name}.")
            summary[ds_path.name] = {}
            summary[ds_path.name]["static"] = static_metrics
            summary[ds_path.name]["avg"] = {}

            for key, values in avg_metrics.items():
                if key == "per_class_aa":
                    # Handle nested dict for per-class AA
                    summary[ds_path.name]["avg"]["per_class_aa"] = {}
                    for class_name, class_values in values.items():
                        if class_values:
                            summary[ds_path.name]["avg"]["per_class_aa"][class_name] = {
                                "mean": statistics.mean(class_values),
                                "std": statistics.stdev(class_values) if len(class_values) > 1 else 0.0
                            }
                elif values:
                    # Handle flat list metrics
                    summary[ds_path.name]["avg"][key] = {
                        "mean": statistics.mean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0.0
                    }
        else:
            print(f"  No valid seed data found for {ds_path.name}. No summary will be generated.")

    # --- Generate Output Files ---
    if not summary:
        print("\nNo data was summarized. Exiting without creating report files.")
        return

    output_dir = Path("reports/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save summary as CSV
    csv_path = output_dir / "metrics_summary_full.csv"
    try:
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow([
                "Dataset", "OA (%)", "AA (%)", "Kappa (%)", 
                "Train Time (min)", "Infer Time (s)", "Model Size (MB)", "Parameters (M)"
            ])
            # Write data rows
            for ds, metrics in summary.items():
                avg_m = metrics.get("avg", {})
                static_m = metrics.get("static", {})
                
                oa_mean = avg_m.get("oa", {}).get("mean", 0)
                oa_std = avg_m.get("oa", {}).get("std", 0)
                aa_mean = avg_m.get("aa", {}).get("mean", 0)
                aa_std = avg_m.get("aa", {}).get("std", 0)
                kappa_mean = avg_m.get("kappa", {}).get("mean", 0)
                kappa_std = avg_m.get("kappa", {}).get("std", 0)
                train_time_mean = avg_m.get("train_time", {}).get("mean", 0)
                train_time_std = avg_m.get("train_time", {}).get("std", 0)
                infer_mean = avg_m.get("infer_time_total", {}).get("mean", 0)
                infer_std = avg_m.get("infer_time_total", {}).get("std", 0)
                
                model_size = static_m.get("model_size", 0)
                num_params = static_m.get("num_params", 0)

                writer.writerow([
                    ds,
                    f"{oa_mean*100:.2f} ± {oa_std*100:.2f}",
                    f"{aa_mean*100:.2f} ± {aa_std*100:.2f}",
                    f"{kappa_mean*100:.2f} ± {kappa_std*100:.2f}",
                    f"{train_time_mean:.2f} ± {train_time_std:.2f}",
                    f"{infer_mean:.2f} ± {infer_std:.2f}",
                    f"{model_size:.2f}",  # No std dev
                    f"{num_params/1e6:.2f}M", # No std dev, add M for Millions
                ])
        print(f"\n✅ Full summary table saved to {csv_path}")
    except IOError as e:
        print(f"\n❌ Error writing CSV file: {e}")

    # 2. Save summary in clean text format
    txt_path = output_dir / "metrics_summary_full.txt"
    try:
        with open(txt_path, "w") as f:
            f.write(f"{'Dataset':<20} {'OA (%)':>18} {'AA (%)':>18} {'Kappa (%)':>18}\n")
            f.write("-" * 75 + "\n")
            for ds, metrics in summary.items():
                avg_m = metrics.get("avg", {})
                f.write(
                    f"{ds:<20} "
                    f"{avg_m.get('oa', {}).get('mean', 0)*100:7.2f} ± {avg_m.get('oa', {}).get('std', 0)*100:4.2f}      "
                    f"{avg_m.get('aa', {}).get('mean', 0)*100:7.2f} ± {avg_m.get('aa', {}).get('std', 0)*100:4.2f}      "
                    f"{avg_m.get('kappa', {}).get('mean', 0)*100:7.2f} ± {avg_m.get('kappa', {}).get('std', 0)*100:4.2f}\n"
                )
            
            f.write("\n" + "-" * 75 + "\n")
            f.write(f"{'Dataset':<20} {'Train Time (min)':>18} {'Infer Time (s)':>18} {'Params (M)':>18}\n")
            f.write("-" * 75 + "\n")
            for ds, metrics in summary.items():
                avg_m = metrics.get("avg", {})
                static_m = metrics.get("static", {})
                f.write(
                    f"{ds:<20} "
                    f"{avg_m.get('train_time', {}).get('mean', 0):7.2f} ± {avg_m.get('train_time', {}).get('std', 0):4.2f}      "
                    f"{avg_m.get('infer_time_total', {}).get('mean', 0):7.2f} ± {avg_m.get('infer_time_total', {}).get('std', 0):4.2f}      "
                    f"{static_m.get('num_params', 0)/1e6:7.2f}M\n" # No std dev
                )

            # 3. Add per-class accuracy report
            f.write("\n\n" + "=" * 75 + "\n")
            f.write("Per-Class Average Accuracies (%)\n")
            f.write("=" * 75 + "\n")
            for ds, metrics in summary.items():
                f.write(f"\n--- {ds} ---\n")
                avg_m = metrics.get("avg", {}).get("per_class_aa", {})
                static_m = metrics.get("static", {})
                class_names = static_m.get("class_names", [])
                
                if not avg_m or not class_names:
                    f.write("  No per-class accuracy data found.\n")
                    continue
                
                # Find max class name length for formatting
                max_len = max(len(name) for name in class_names) + 2
                
                for class_name in class_names:
                    class_stats = avg_m.get(class_name, {})
                    class_mean = class_stats.get("mean", 0)
                    class_std = class_stats.get("std", 0)
                    f.write(f"  {class_name:<{max_len}} {class_mean*100:7.2f} ± {class_std*100:4.2f}\n")

        print(f"✅ Clean text summary (with per-class) saved to {txt_path}")
    except IOError as e:
        print(f"\n❌ Error writing TXT file: {e}")


    # 4. Save summary in LaTeX table format
    latex_path = output_dir / "metrics_summary_latex.txt"
    try:
        with open(latex_path, "w") as f:
            f.write("% --- Main Performance Table --- \n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Performance and Efficiency Metrics}\n")
            f.write("\\label{tab:results}\n")
            f.write("\\resizebox{\\textwidth}{!}{\n") # Make table fit width
            f.write("\\begin{tabular}{lccccc}\n")
            f.write("\\toprule\n")
            f.write("Dataset & OA (\%) & AA (\%) & Kappa (\%) & Train Time (min) & Params (M) \\\\\n")
            f.write("\\midrule\n")
            for ds, metrics in summary.items():
                avg_m = metrics.get("avg", {})
                static_m = metrics.get("static", {})
                ds_fmt = ds.replace("_", " ") # Format dataset name
                
                oa_str = f"{avg_m.get('oa', {}).get('mean', 0)*100:.2f} $\\pm$ {avg_m.get('oa', {}).get('std', 0)*100:.2f}"
                aa_str = f"{avg_m.get('aa', {}).get('mean', 0)*100:.2f} $\\pm$ {avg_m.get('aa', {}).get('std', 0)*100:.2f}"
                kappa_str = f"{avg_m.get('kappa', {}).get('mean', 0)*100:.2f} $\\pm$ {avg_m.get('kappa', {}).get('std', 0)*100:.2f}"
                train_time_str = f"{avg_m.get('train_time', {}).get('mean', 0):.2f} $\\pm$ {avg_m.get('train_time', {}).get('std', 0):.2f}"
                params_str = f"{static_m.get('num_params', 0)/1e6:.2f}" # No std dev
                
                f.write(f"{ds_fmt} & {oa_str} & {aa_str} & {kappa_str} & {train_time_str} & {params_str} \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("}\n") # End resizebox
            f.write("\\end{table}\n\n")

            # --- Per-class accuracy tables ---
            f.write("% --- Per-Class Accuracy Tables --- \n")
            for ds, metrics in summary.items():
                avg_m = metrics.get("avg", {}).get("per_class_aa", {})
                static_m = metrics.get("static", {})
                class_names = static_m.get("class_names", [])
                ds_fmt = ds.replace("_", " ")

                if not avg_m or not class_names:
                    continue

                f.write(f"\n% Per-class results for {ds_fmt}\n")
                f.write("\\begin{table}[h]\n")
                f.write("\\centering\n")
                f.write(f"\\caption{{Per-Class Accuracy (\%) for {ds_fmt}}}\n")
                f.write(f"\\label{{tab:pca_{ds.lower()}}}\n")
                
                # Split classes into two columns if too many
                num_classes = len(class_names)
                mid_point = (num_classes + 1) // 2
                
                f.write("\\begin{tabular}{lc@{\\hspace{4em}}lc}\n") # l c @{...} l c
                f.write("\\toprule\n")
                f.write("Class & Accuracy (\%) & Class & Accuracy (\%) \\\\\n")
                f.write("\\midrule\n")
                
                for i in range(mid_point):
                    # Left column
                    class_1 = class_names[i]
                    stats_1 = avg_m.get(class_1, {})
                    mean_1 = stats_1.get("mean", 0) * 100
                    std_1 = stats_1.get("std", 0) * 100
                    acc_str_1 = f"{mean_1:.2f} $\\pm$ {std_1:.2f}"
                    
                    f.write(f"{class_1.replace('_', ' ')} & {acc_str_1}")
                    
                    # Right column (if it exists)
                    if i + mid_point < num_classes:
                        class_2 = class_names[i + mid_point]
                        stats_2 = avg_m.get(class_2, {})
                        mean_2 = stats_2.get("mean", 0) * 100
                        std_2 = stats_2.get("std", 0) * 100
                        acc_str_2 = f"{mean_2:.2f} $\\pm$ {std_2:.2f}"
                        f.write(f" & {class_2.replace('_', ' ')} & {acc_str_2} \\\\\n")
                    else:
                        f.write(" & & \\\\\n") # Empty right column
                        
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n")

        print(f"✅ LaTeX summary tables saved to {latex_path}")
    except IOError as e:
        print(f"\n❌ Error writing LaTeX file: {e}")


if __name__ == "__main__":
    # Ensure the root directory is correct
    # This assumes the script is run from the project root (e.g., .../SMBFT/)
    # And the models are in .../SMBFT/models/final/smbft
    # If you run from .../SMBFT/scripts/, change this path.
    
    # Path from the script's location (assuming script is in project root)
    script_dir = Path(__file__).resolve().parent
    models_root = script_dir / "models" / "final" / "smbft"
    
    # Check if the target directory exists
    if not models_root.is_dir():
        print(f"Error: Directory not found: {models_root}")
        print("Please run this script from the project's root directory,")
        print("or adjust the 'models_root' path inside the script.")
    else:
        summarize_metrics(str(models_root))