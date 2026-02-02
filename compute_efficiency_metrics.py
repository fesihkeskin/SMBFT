#!/usr/bin/env python3
"""compute_efficiency_metrics.py

Collects efficiency metrics for SMBFT across datasets:
- Number of parameters (M)
- MACs/GFLOPs per patch
- Train time per epoch (s)
- Inference time per patch (ms)

Reads from saved train_history_*.json and test metrics_*.json files,
and calculates computational complexity using fvcore or thop.

Usage:
    python compute_efficiency_metrics.py --dataset Houston13 --checkpoint models/final/smbft/Houston13/smbft_best_Houston13.pth
    python compute_efficiency_metrics.py --all  # Process all datasets in models/final/smbft/

Outputs:
    - efficiency_metrics.csv
    - efficiency_metrics_latex.tex
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_architecture import SMBFT
from src.data.data_loader import load_dataset


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_macs_fvcore(model: nn.Module, input_shape: Tuple[int, ...], device: str = "cpu") -> Optional[float]:
    """
    Compute MACs using fvcore.nn.FlopCountAnalysis.
    Returns MACs in billions (GMACs), or None if fvcore is not available.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        model = model.to(device)
        model.eval()
        dummy_input = torch.randn(1, *input_shape, device=device)
        with torch.no_grad():
            flops = FlopCountAnalysis(model, dummy_input)
            total_flops = flops.total()
        # FLOPs ≈ 2 * MACs (1 MAC = 1 multiply + 1 add = 2 FLOPs)
        # Return GMACs
        return total_flops / 2e9
    except ImportError:
        return None
    except Exception as e:
        print(f"Warning: fvcore profiling failed ({e})")
        return None


def compute_macs_thop(model: nn.Module, input_shape: Tuple[int, ...], device: str = "cpu") -> Optional[float]:
    """
    Compute MACs using thop (torch-ops-counter).
    Returns MACs in billions (GMACs), or None if thop is not available.
    """
    try:
        from thop import profile
        model = model.to(device)
        model.eval()
        dummy_input = torch.randn(1, *input_shape, device=device)
        with torch.no_grad():
            macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        # thop returns MACs directly
        return macs / 1e9
    except ImportError:
        return None
    except Exception as e:
        print(f"Warning: thop profiling failed ({e})")
        return None


def compute_macs_ptflops(model: nn.Module, input_shape: Tuple[int, ...], device: str = "cpu") -> Optional[float]:
    """
    Compute MACs using ptflops.
    Returns MACs in billions (GMACs), or None if ptflops is not available.
    """
    try:
        from ptflops import get_model_complexity_info
        model = model.to(device)
        model.eval()
        # ptflops expects (C, H, W) for images
        macs, params = get_model_complexity_info(
            model, input_shape, as_strings=False,
            print_per_layer_stat=False, verbose=False
        )
        # ptflops returns MACs
        return macs / 1e9
    except ImportError:
        return None
    except Exception as e:
        print(f"Warning: ptflops profiling failed ({e})")
        return None


def compute_macs(model: nn.Module, input_shape: Tuple[int, ...], device: str = "cpu") -> Tuple[Optional[float], str]:
    """
    Try multiple MAC computation methods in order of preference.
    Returns (GMACs, method_name) or (None, "unavailable").
    """
    # Try fvcore first (most reliable for transformers)
    gmacs = compute_macs_fvcore(model, input_shape, device)
    if gmacs is not None:
        return gmacs, "fvcore"
    
    # Try thop
    gmacs = compute_macs_thop(model, input_shape, device)
    if gmacs is not None:
        return gmacs, "thop"
    
    # Try ptflops
    gmacs = compute_macs_ptflops(model, input_shape, device)
    if gmacs is not None:
        return gmacs, "ptflops"
    
    return None, "unavailable"


def _infer_bands_classes_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[Optional[int], Optional[int]]:
    """Try to infer (in_bands, n_classes) from known layer names."""
    n_classes = None
    in_bands = None

    # Classification head: Linear(spec_dim -> n_classes)
    if "classification_head.1.weight" in state_dict:
        n_classes = state_dict["classification_head.1.weight"].shape[0]
    elif "classification_head.1.bias" in state_dict:
        n_classes = state_dict["classification_head.1.bias"].shape[0]

    # Reconstruction head: Linear(dim_feedforward -> in_bands)
    if "reconstruction_head.2.weight" in state_dict:
        in_bands = state_dict["reconstruction_head.2.weight"].shape[0]
    elif "reconstruction_head.2.bias" in state_dict:
        in_bands = state_dict["reconstruction_head.2.bias"].shape[0]

    return in_bands, n_classes


def _infer_bands_classes_from_dataset(dataset_name: str) -> Tuple[int, int]:
    """Infer (in_bands, n_classes) from dataset files."""
    cube, gt = load_dataset(dataset_name)
    in_bands = cube.shape[2]
    n_classes = len(np.unique(gt)) - 1  # ignore background label 0
    return in_bands, n_classes


def load_checkpoint_and_build_model(
    checkpoint_path: Path,
    dataset_name: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[nn.Module, dict]:
    """Load checkpoint and build model from saved args."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "args" not in ckpt:
        raise KeyError(f"Checkpoint {checkpoint_path} missing 'args' dict.")
    
    args_dict = ckpt["args"]
    
    # Extract model hyperparameters
    n_classes = args_dict.get("n_classes")
    in_bands = args_dict.get("in_bands")
    
    # If not in checkpoint, try to infer from model state dict
    if n_classes is None or in_bands is None:
        state_dict = ckpt.get("model_state_dict", ckpt)
        inferred_bands, inferred_classes = _infer_bands_classes_from_state_dict(state_dict)
        if in_bands is None:
            in_bands = inferred_bands
        if n_classes is None:
            n_classes = inferred_classes
    
    # Final fallback: infer from dataset files if available
    if (n_classes is None or in_bands is None) and dataset_name:
        try:
            inferred_bands, inferred_classes = _infer_bands_classes_from_dataset(dataset_name)
            if in_bands is None:
                in_bands = inferred_bands
            if n_classes is None:
                n_classes = inferred_classes
        except Exception as e:
            print(f"Warning: Could not infer from dataset '{dataset_name}' ({e})")

    if n_classes is None or in_bands is None:
        raise ValueError("Cannot determine n_classes or in_bands from checkpoint or dataset.")
    
    model = SMBFT(
        in_bands=in_bands,
        patch_size=args_dict.get("patch_size", 11),
        n_classes=n_classes,
        spec_dim=args_dict.get("spec_dim", 64),
        spat_dim=args_dict.get("spat_dim", 64),
        n_heads=args_dict.get("n_heads", 4),
        n_layers=args_dict.get("n_layers", 4),
        fusion=args_dict.get("fusion", "graph"),
        mask_ratio=args_dict.get("mask_ratio", 0.2),
        mask_mode=args_dict.get("mask_mode", "random"),
        dim_feedforward=args_dict.get("dim_feedforward", 512),
        dropout=args_dict.get("dropout", 0.2),
        d3_out_channels=args_dict.get("d3_out_channels", 16),
        use_cross_attn=not args_dict.get("no_cross_attn", False),
    )
    
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    
    return model, args_dict


def extract_metrics_from_jsons(
    dataset_dir: Path, dataset_name: str
) -> Dict[str, float]:
    """Extract metrics from train_history and test metrics JSONs."""
    metrics = {}
    
    # Find train_history JSON
    train_history_files = list(dataset_dir.glob("train_history_*.json"))
    if train_history_files:
        with open(train_history_files[0], "r") as f:
            train_data = json.load(f)
        
        # Parameters
        metrics["num_parameters"] = train_data.get("num_parameters", 0)
        
        # Train time per epoch
        train_time_min = train_data.get("train_time_minutes", 0)
        num_epochs = len(train_data.get("epoch_metrics", []))
        if num_epochs > 0:
            metrics["train_time_per_epoch_sec"] = (train_time_min * 60) / num_epochs
        else:
            metrics["train_time_per_epoch_sec"] = 0
    
    # Find test metrics JSON
    test_results_dir = dataset_dir / "test_results"
    if test_results_dir.exists():
        metrics_files = list(test_results_dir.glob("metrics_seed_*.json"))
        if metrics_files:
            with open(metrics_files[0], "r") as f:
                test_data = json.load(f)
            
            # Inference time per sample in ms
            infer_time_sec = test_data.get("Inference Time Per Sample (seconds)", 0)
            metrics["inference_time_ms_per_patch"] = infer_time_sec * 1000
    
    return metrics


def process_dataset(
    dataset_name: str,
    checkpoint_path: Path,
    device: str = "cpu"
) -> Dict[str, any]:
    """Process a single dataset and return efficiency metrics."""
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*60}")
    
    # Load model
    model, args_dict = load_checkpoint_and_build_model(checkpoint_path, dataset_name, device)
    
    # Get input shape (C, H, W) for a single patch
    in_bands = args_dict.get("in_bands")
    if in_bands is None:
        in_bands = getattr(model, "in_bands", None)
    patch_size = args_dict.get("patch_size", 11)
    input_shape = (in_bands, patch_size, patch_size)
    
    # Count parameters
    params_M = count_parameters(model) / 1e6
    
    # Compute MACs
    gmacs, method = compute_macs(model, input_shape, device)
    
    # Extract metrics from JSONs
    dataset_dir = checkpoint_path.parent
    json_metrics = extract_metrics_from_jsons(dataset_dir, dataset_name)
    
    result = {
        "dataset": dataset_name,
        "params_M": params_M,
        "gmacs": gmacs,
        "macs_method": method,
        "train_time_per_epoch_sec": json_metrics.get("train_time_per_epoch_sec", None),
        "inference_time_ms_per_patch": json_metrics.get("inference_time_ms_per_patch", None),
        "patch_size": patch_size,
        "in_bands": in_bands,
    }
    
    print(f"Parameters: {params_M:.2f}M")
    print(f"GMACs (per patch): {gmacs:.3f} [{method}]" if gmacs else "GMACs: N/A")
    print(f"Train time/epoch: {result['train_time_per_epoch_sec']:.2f}s" if result['train_time_per_epoch_sec'] else "Train time/epoch: N/A")
    print(f"Inference: {result['inference_time_ms_per_patch']:.3f}ms/patch" if result['inference_time_ms_per_patch'] else "Inference: N/A")
    
    return result


def generate_latex_table(results: List[Dict], output_path: Path):
    """Generate LaTeX table from results."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Efficiency of SMBFT (report hardware + implementation details).}")
    lines.append(r"\label{tab:efficiency}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Params (M) & GMACs (per patch) & Train time/epoch (s) & Inference (ms/patch) \\")
    lines.append(r"\midrule")
    
    for r in results:
        dataset = r["dataset"].replace("_", " ")
        params = f"{r['params_M']:.2f}" if r['params_M'] else "N/A"
        gmacs = f"{r['gmacs']:.3f}" if r['gmacs'] else "N/A"
        train_time = f"{r['train_time_per_epoch_sec']:.1f}" if r['train_time_per_epoch_sec'] else "N/A"
        inference = f"{r['inference_time_ms_per_patch']:.3f}" if r['inference_time_ms_per_patch'] else "N/A"
        
        lines.append(f"{dataset} & {params} & {gmacs} & {train_time} & {inference} \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{2pt}")
    lines.append(r"\footnotesize{\textbf{Hardware:} [TODO: Fill in GPU/CPU model, batch size, AMP on/off].}")
    lines.append(r"\end{table}")
    
    output_path.write_text("\n".join(lines))
    print(f"\n✅ LaTeX table saved to: {output_path}")


def generate_csv(results: List[Dict], output_path: Path):
    """Generate CSV from results."""
    import csv
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "params_M", "gmacs", "macs_method",
            "train_time_per_epoch_sec", "inference_time_ms_per_patch",
            "patch_size", "in_bands"
        ])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ CSV saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute SMBFT efficiency metrics.")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name (e.g., Houston13, Pavia_University, Salinas)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint file."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all datasets in models/final/smbft/."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/results",
        help="Output directory for results."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for profiling (cpu or cuda)."
    )
    
    args = parser.parse_args()
    
    results = []
    
    if args.all:
        # Process all datasets in models/final/smbft/
        base_dir = PROJECT_ROOT / "models" / "final" / "smbft"
        if not base_dir.exists():
            print(f"ERROR: Directory not found: {base_dir}")
            return 1
        
        for dataset_dir in sorted(base_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            
            dataset_name = dataset_dir.name
            # Find checkpoint
            ckpt_files = list(dataset_dir.glob("smbft_best_*.pth"))
            if not ckpt_files:
                print(f"Warning: No checkpoint found for {dataset_name}, skipping.")
                continue
            
            checkpoint_path = ckpt_files[0]
            
            try:
                result = process_dataset(dataset_name, checkpoint_path, args.device)
                results.append(result)
            except Exception as e:
                print(f"ERROR processing {dataset_name}: {e}")
    
    else:
        # Process single dataset
        if not args.dataset or not args.checkpoint:
            print("ERROR: --dataset and --checkpoint are required (or use --all).")
            return 1
        
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"ERROR: Checkpoint not found: {checkpoint_path}")
            return 1
        
        try:
            result = process_dataset(args.dataset, checkpoint_path, args.device)
            results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            return 1
    
    if not results:
        print("No results to save.")
        return 1
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_csv(results, output_dir / "efficiency_metrics.csv")
    generate_latex_table(results, output_dir / "efficiency_metrics_latex.tex")
    
    print("\n" + "="*60)
    print("✅ Efficiency metrics computation complete!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Usage:
# Process all datasets
# python compute_efficiency_metrics.py --all

# Process single dataset
# python compute_efficiency_metrics.py --dataset Houston13 --checkpoint models/final/smbft/Houston13/smbft_best_Houston13.pth