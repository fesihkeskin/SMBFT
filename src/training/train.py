#!/usr/bin/env python3
# src/training/train.py

"""
End-to-end training routine for SMBFT with stratified splitting,
Optuna pruning, and advanced checkpointing with a tie-breaker mechanism.
"""

import os
import time
import gc
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from optuna.exceptions import TrialPruned
from torch.amp import autocast, GradScaler

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from src.data.dataset import HyperspectralDataset
from src.models.model_architecture import SMBFT
from src.utils.utils import save_checkpoint, stratified_min_samples_split, FocalLoss, fit_pca_from_pixels, apply_pca_with_state
from src.data.dataset_info import get_imbalance_ratio

def _load_split(args: Namespace, stage: str, indices, mean, std):
    """Helper to create a HyperspectralDataset instance."""
    return HyperspectralDataset(
        cube_path=str(args.cube_path), gt_path=str(args.gt_path),
        patch_size=args.patch_size, mode=stage, indices=indices,
        mean=mean, std=std, augment=(stage == "train"), cfg=args,
        # Pass the in-memory data if available using the correct keyword
        preloaded_cube=getattr(args, "raw_cube", None)
    )

def _inv_freq_weights(labels: np.ndarray, n_cls: int, device: torch.device):
    """Computes inverse frequency weights for classes."""
    valid_labels = labels[labels >= 0]
    if len(valid_labels) == 0:
        return torch.ones(n_cls, dtype=torch.float32, device=device)
    cnt = np.bincount(valid_labels, minlength=n_cls)
    weights = 1.0 / np.log1p(np.maximum(cnt, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """Applies MixUp to a batch of data."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return x_mix, y_a, y_b, lam

def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """Applies CutMix to a batch of data."""
    lam = np.random.beta(alpha, alpha)
    _, _, H, W = x.size()
    index = torch.randperm(x.size(0), device=x.device)
    cut_ratio = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_ratio), int(H * cut_ratio)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def get_model_size(model):
    """Returns the model size in MB."""
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        torch.save(model.state_dict(), tmp.name)
        size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
    os.remove(tmp.name)
    return size_mb

def train(args: Namespace, trial=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(f"🚀 Training on device: {device}")

    # ── 1. Print Hyperparameters ──────────────────────────────────────────
    print("🔧 Hyperparameter settings:")
    hp_dict = {k: v for k, v in sorted(vars(args).items()) if not isinstance(v, (np.ndarray, Path))}
    print(json.dumps(hp_dict, indent=4))

    # ── 2. Stratified Split and Index Saving ──────────────────────────────
    coords = np.argwhere(args.raw_gt > 0)
    labels = args.raw_gt[coords[:, 0], coords[:, 1]] - 1
    train_idx, val_idx, test_idx = stratified_min_samples_split(
        coords, labels, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
    )
    split_dir = Path(args.output_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / f"train_idx_seed_{args.seed}.npy", train_idx)
    np.save(split_dir / f"val_idx_seed_{args.seed}.npy", val_idx)
    np.save(split_dir / f"test_idx_seed_{args.seed}.npy", test_idx)
    print(f"📊 Saved data split indices to {split_dir.resolve()}")

    # ── 2.1 Optional PCA (fit on TRAIN pixels; reuse later) ───────────────
    if getattr(args, "pca_components", 0) and args.pca_components > 0:
        print(f"Applying PCA, reducing bands to {args.pca_components} (fit on train split)...")
        pca_mean, pca_comps = fit_pca_from_pixels(args.raw_cube, train_idx, args.pca_components)
        args.pca_state = {
            "mean": pca_mean.tolist(),
            "components": pca_comps.tolist(),
        }
        args.raw_cube = apply_pca_with_state(args.raw_cube, pca_mean, pca_comps)
        print(f"✅ PCA done. New shape: {args.raw_cube.shape}")

    # ── 3. Build Datasets and Loaders ───────────────────────────────────────
    train_ds = _load_split(args, "train", train_idx, None, None)
    val_ds = _load_split(args, "val", val_idx, train_ds.mean, train_ds.std)

    pin_memory = (device.type == "cuda")
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    # ── 4. Model, Loss, and Optimizer ───────────────────────────────────────
    n_cls = len(np.unique(args.raw_gt)) - 1
    print(f"Number of classes: {n_cls}")
    model = SMBFT(
        in_bands=train_ds.D, patch_size=args.patch_size, n_classes=n_cls,
        spec_dim=args.spec_dim, spat_dim=args.spat_dim, n_heads=args.n_heads,
        n_layers=args.n_layers, fusion=args.fusion,
        mask_ratio=0.0 if args.no_spec_mask else args.mask_ratio,
        mask_mode=args.mask_mode, dim_feedforward=args.dim_feedforward,
        dropout=args.dropout, d3_out_channels=args.d3_out_channels,
        use_cross_attn=not getattr(args, "no_cross_attn", False),
    ).to(device)

    crit_recon = nn.MSELoss()
    imbalance_ratio = get_imbalance_ratio(train_ds.labels)
    print(f"\n⚖️ Train set imbalance ratio: {imbalance_ratio:.2f}")

    if imbalance_ratio > 5.0:
        print("⚠️ Dataset is imbalanced. Using FocalLoss.")
        class_weights = _inv_freq_weights(train_ds.labels, n_cls, device)
        crit_cls = FocalLoss(gamma=2.0, alpha=class_weights, ignore_index=-1).to(device)
    else:
        print("✅ Dataset is balanced. Using CrossEntropyLoss with label smoothing.")
        crit_cls = nn.CrossEntropyLoss(
            weight=_inv_freq_weights(train_ds.labels, n_cls, device),
            label_smoothing=0.05, ignore_index=-1
        ).to(device)

    optimiser = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, "max", patience=10, factor=0.3, verbose=True)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    print(f"▶ Train samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_parameters:,}")

    best_val_oa, patience_counter = 0.0, 0
    best_train_acc_at_max_oa = 0.0  # Tracks train acc for tie-breaking
    epoch_history = []
    t0 = time.time()

    # ── 5. Training Loop ────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, cls_loss_sum, recon_loss_sum, total_correct, total_samples = 0,0,0,0,0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False)
        for batch in pbar:
            optimiser.zero_grad(set_to_none=True)
            x, y = batch["patch"].to(device), batch["label"].to(device)
            B, C, _, _ = x.shape

            x_aug, y_a, y_b, lam = x, y, y, 1.0
            if not args.no_mixup and np.random.rand() < 0.5:
                x_aug, y_a, y_b, lam = mixup_data(x, y, alpha=0.4)
            elif not args.no_cutmix and np.random.rand() < 0.5:
                x_aug, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0)

            with autocast('cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
                if not args.no_spec_mask:
                    logits, rec_tokens, mask_flat = model(x_aug, return_mask=True)
                    loss_cls = lam * crit_cls(logits, y_a) + (1 - lam) * crit_cls(logits, y_b)
                    spec_flat_target = x.view(B, C, -1).permute(2, 0, 1)
                    if mask_flat is not None and mask_flat.any():
                        loss_recon = crit_recon(rec_tokens[mask_flat], spec_flat_target[mask_flat])
                        loss = loss_cls + args.recon_weight * loss_recon
                        recon_loss_sum += loss_recon.item() * B
                    else:
                        loss = loss_cls
                else:
                    logits = model(x_aug, return_mask=False)
                    loss_cls = lam * crit_cls(logits, y_a) + (1 - lam) * crit_cls(logits, y_b)
                    loss = loss_cls

            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            total_loss += loss.item() * B
            cls_loss_sum += loss_cls.item() * B
            total_correct += (logits.argmax(1) == y).sum().item()
            total_samples += B
            pbar.set_postfix(loss=f"{total_loss/total_samples:.4f}", acc=f"{total_correct/total_samples:.3f}")

        train_avg_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        val_oa = validate(model, val_loader, device)
        scheduler.step(val_oa)

        print(f"Epoch {epoch:03d} | Train Loss: {train_avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val OA: {val_oa:.4f}")
        epoch_history.append({"epoch": epoch, "train_loss": train_avg_loss, "train_acc": train_acc, "val_oa": val_oa, "lr": optimiser.param_groups[0]["lr"]})

        # --- Advanced Checkpointing with Tie-Breaker ---
        save_model = False
        # Condition 1: Standard improvement in validation accuracy.
        if val_oa > best_val_oa:
            best_val_oa = val_oa
            best_train_acc_at_max_oa = train_acc  # Update tie-breaker metric
            patience_counter = 0
            save_model = True
            print(f"  ✔ New best val OA {best_val_oa:.4f} (checkpoint will be saved)")
        
        # Condition 2: Tie-breaker for perfect (or near-perfect) validation accuracy.
        # If val OA is maxed out, we save the model if its training accuracy is higher.
        elif val_oa >= 0.9999 and val_oa == best_val_oa and train_acc > best_train_acc_at_max_oa:
            best_train_acc_at_max_oa = train_acc # Update to new best train acc
            patience_counter = 0 # Reset patience because the model is still improving
            save_model = True
            print(f"  ✔ Val OA maxed. New best train Acc: {train_acc:.4f} (checkpoint will be saved)")
        
        # No improvement.
        else:
            patience_counter += 1

        if save_model:
            save_checkpoint(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_oa": best_val_oa, "args": vars(args)},
                args.output_dir, f"smbft_best_{args.dataset}.pth"
            )
        
        if patience_counter >= args.early_stop:
            print(f"⏹ Early stop triggered at epoch {epoch}")
            break
        # --- End of Checkpointing Logic ---

        if trial is not None:
            trial.report(val_oa, epoch)
            if trial.should_prune():
                print(f"⏹️ Trial pruned by Optuna at epoch {epoch}")
                raise TrialPruned()

    # ── 6. Final Test on Best Model ─────────────────────────────────────────
    print("\n--- Final Evaluation on Test Set ---")
    ckpt_path = Path(args.output_dir) / f"smbft_best_{args.dataset}.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_ds = _load_split(args, "test", test_idx, train_ds.mean, train_ds.std)
        test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
        test_oa = validate(model, test_loader, device)
        print(f"🏁 Finished. Best Val OA: {best_val_oa:.4f} | Final Test OA: {test_oa:.4f} | Time: {(time.time()-t0)/60:.1f} min")
    else:
        test_oa = -1.0
        print("🏁 No checkpoint found. Final test skipped.")

    # ── 7. Save History JSON ──────────────────────────────────────────────
    train_time_minutes = round((time.time() - t0) / 60, 2)
    history = {
        "dataset": args.dataset,
        "hyperparameters": hp_dict,
        "epoch_metrics": epoch_history,
        "best_val_OA": best_val_oa,
        "test_OA": test_oa,
        "train_time_minutes": train_time_minutes,
        "Model_size_MB": f"{get_model_size(model):.2f} MB",
        "num_parameters": num_parameters
    }
    history_path = Path(args.output_dir) / f"train_history_{args.dataset}_seed{args.seed}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"📜 Training history saved to {history_path.resolve()}")

    del model, optimiser, crit_cls, crit_recon
    gc.collect()
    torch.cuda.empty_cache()
    return best_val_oa

def train_with_oom_handling(args, trial):
    try:
        return train(args, trial)
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print("⚠️ OOM: pruning trial / reduce batch size.")
            torch.cuda.empty_cache()
            raise TrialPruned()
        else:
            raise

def validate(model, val_loader, device):
    """Calculates Overall Accuracy for a given model and dataloader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch["patch"].to(device), batch["label"].to(device)
            logits = model(x)
            preds = logits.argmax(1)
            mask = (y != -1)
            correct += (preds[mask] == y[mask]).sum().item()
            total += mask.sum().item()
    return correct / total if total > 0 else 0.0