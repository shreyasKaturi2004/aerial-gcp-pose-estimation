"""
train.py — Training loop for GCPNet.

Loss function:
    total_loss = lambda_reg * SmoothL1(coords_pred, coords_true)
               + CrossEntropy(logits, class_true, weight=class_weights)

Why SmoothL1 (Huber) over MSE:
  - Transitions from squared error (for small residuals) to linear (for large
    residuals). This reduces the influence of any imprecisely labelled samples
    without throwing away their contribution entirely.

Why lambda_reg=10:
  - At random initialisation, SmoothL1 on [0,1] coordinates is ~0.05–0.1,
    while cross-entropy for 3 classes is ~1.1. Without scaling, the cls loss
    dominates and the network almost ignores localisation. lambda_reg=10 brings
    both losses to the same order of magnitude.

Why class-weighted cross-entropy:
  - Square is 2.7× more frequent than Cross in the training set. Without
    weighting, the classifier can reach ~50% accuracy by predicting Square
    for everything. Weights ∝ 1/frequency push the model to learn all three.

Why cosine annealing LR scheduler:
  - Smoothly decays the learning rate to near-zero over training. Works well
    with Adam for fine-tuning pretrained backbones — avoids the sharp drops of
    step-decay that can destabilise training on small datasets.

PCK metric:
  - Evaluated in the ORIGINAL pixel space (before any resize). The model
    predicts x_norm = x/W_orig, y_norm = y/H_orig, which are multiplied back
    by the original image dimensions to compute pixel-space distances.
    Thresholds: 10px, 25px, 50px (in original 4096-px image space).

Usage:
    python train.py
    python train.py --epochs 60 --batch-size 8 --lr 5e-5
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from dataset import (
    CLASS_NAMES,
    GCPDataset,
    IMG_SIZE,
    load_labels,
    make_splits,
    split_summary,
)
from model import GCPNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = Path("checkpoints")


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_class_weights(paths, labels) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss, normalised to n_classes."""
    counts = Counter(labels[p]["verified_shape"] for p in paths)
    total  = sum(counts.values())
    n      = len(CLASS_NAMES)
    weights = [total / (n * counts[cls]) for cls in CLASS_NAMES]
    return torch.tensor(weights, dtype=torch.float32)


def pck_metric(
    pred_norm: torch.Tensor,
    true_norm: torch.Tensor,
    orig_wh: torch.Tensor,
    threshold_px: float,
) -> float:
    """
    Percentage of Correct Keypoints within `threshold_px` pixels.

    pred_norm, true_norm : (N, 2) in [0, 1] — [x_norm, y_norm]
    orig_wh              : (N, 2) — [W_orig, H_orig] per sample
    """
    pred_px = pred_norm * orig_wh   # (N, 2), pixel coords in original image
    true_px = true_norm * orig_wh
    dist    = torch.norm(pred_px - true_px, dim=1)   # (N,) Euclidean pixel dist
    return (dist <= threshold_px).float().mean().item()


# ── Epoch runner ──────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    reg_criterion: nn.Module,
    cls_criterion: nn.Module,
    lambda_reg: float,
    train: bool,
) -> dict:
    model.train(train)

    total_loss = total_reg = total_cls = 0.0
    pred_coords_list, true_coords_list, orig_wh_list = [], [], []
    pred_cls_list, true_cls_list = [], []

    first_batch = True
    with torch.set_grad_enabled(train):
        for imgs, coords, classes, orig_wh in loader:
            imgs    = imgs.to(DEVICE)
            coords  = coords.to(DEVICE)
            classes = classes.to(DEVICE)

            pred_coords, logits = model(imgs)

            loss_reg = reg_criterion(pred_coords, coords)
            loss_cls = cls_criterion(logits, classes)
            loss     = lambda_reg * loss_reg + loss_cls

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Print GPU memory usage after the very first training batch
            if first_batch and train and DEVICE.type == "cuda":
                alloc  = torch.cuda.memory_allocated(DEVICE) / 1e9
                reserv = torch.cuda.memory_reserved(DEVICE) / 1e9
                print(f"  [GPU] batch={tuple(imgs.shape)}  "
                      f"allocated={alloc:.2f}GB  reserved={reserv:.2f}GB  "
                      f"total={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
            first_batch = False

            total_loss += loss.item()
            total_reg  += loss_reg.item()
            total_cls  += loss_cls.item()

            pred_coords_list.append(pred_coords.detach().cpu())
            true_coords_list.append(coords.detach().cpu())
            orig_wh_list.append(orig_wh.cpu())
            pred_cls_list.append(logits.argmax(dim=1).detach().cpu())
            true_cls_list.append(classes.cpu())

    n = len(loader)
    pred_all  = torch.cat(pred_coords_list)
    true_all  = torch.cat(true_coords_list)
    wh_all    = torch.cat(orig_wh_list)
    pred_cls  = torch.cat(pred_cls_list).numpy()
    true_cls  = torch.cat(true_cls_list).numpy()

    return {
        "loss":     total_loss / n,
        "reg_loss": total_reg  / n,
        "cls_loss": total_cls  / n,
        "pck10":    pck_metric(pred_all, true_all, wh_all, 10),
        "pck25":    pck_metric(pred_all, true_all, wh_all, 25),
        "pck50":    pck_metric(pred_all, true_all, wh_all, 50),
        "macro_f1": f1_score(true_cls, pred_cls, average="macro", zero_division=0),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    print(f"Device: {DEVICE}")
    CKPT_DIR.mkdir(exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    labels = load_labels()
    train_paths, val_paths = make_splits(labels, val_frac=0.2, seed=42)

    print(f"\nSplit summary (seed=42, val_frac=0.2, split by marker ID):")
    split_summary(labels, train_paths, val_paths)

    train_ds = GCPDataset(train_paths, labels, img_size=args.img_size, augment=True)
    val_ds   = GCPDataset(val_paths,   labels, img_size=args.img_size, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(DEVICE.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GCPNet(pretrained=True).to(DEVICE)

    # ── Loss ──────────────────────────────────────────────────────────────────
    class_weights = compute_class_weights(train_paths, labels).to(DEVICE)
    print(f"\nClass weights (Cross / L-Shape / Square): "
          + " / ".join(f"{w:.3f}" for w in class_weights.cpu()))

    reg_criterion = nn.SmoothL1Loss()
    cls_criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    # Two-group LR: backbone (pretrained) gets 10x lower LR than the new heads.
    # This avoids catastrophically overwriting useful pretrained features.
    backbone_params = list(model.backbone.parameters())
    head_params     = list(model.reg_head.parameters()) + list(model.cls_head.parameters())
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": args.lr / 10},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # ── Resume from checkpoint (if --resume provided) ─────────────────────────
    start_epoch   = 1
    best_val_loss = float("inf")
    best_ckpt_path = CKPT_DIR / "best_model.pth"

    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"\nResumed from epoch {start_epoch - 1}  "
              f"(best val_loss so far: {best_val_loss:.4f})")

    # ── Training loop ─────────────────────────────────────────────────────────
    # Early stopping: halt if PCK@50 + macro_F1 combined hasn't improved in
    # `patience` consecutive epochs. Using the sum of both metrics means
    # training continues as long as EITHER is still improving.
    patience        = args.patience
    no_improve      = 0
    best_combined   = -1.0   # PCK@50 + macro_F1, both in [0,1]

    print(f"\n{'Ep':>4}  {'TrLoss':>8}  {'VaLoss':>8}  {'PCK10':>6}  "
          f"{'PCK25':>6}  {'PCK50':>6}  {'MacF1':>6}  {'NoImp':>5}")
    print("-" * 72)

    for epoch in range(start_epoch, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, reg_criterion,
                       cls_criterion, args.lambda_reg, train=True)
        va = run_epoch(model, val_loader,   optimizer, reg_criterion,
                       cls_criterion, args.lambda_reg, train=False)
        scheduler.step()

        combined = va["pck50"] + va["macro_f1"]
        if combined > best_combined:
            best_combined = combined
            no_improve    = 0
        else:
            no_improve += 1

        # Save checkpoint every epoch (full state for resumability)
        epoch_ckpt = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_loss":        va["loss"],
            "val_metrics":     va,
            "args":            vars(args),
        }
        torch.save(epoch_ckpt, CKPT_DIR / "last_epoch.pth")

        tag = ""
        if va["loss"] < best_val_loss:
            best_val_loss = va["loss"]
            torch.save(epoch_ckpt, best_ckpt_path)
            tag = "  *best*"

        print(f"{epoch:4d}  {tr['loss']:8.4f}  {va['loss']:8.4f}  "
              f"{va['pck10']:6.3f}  {va['pck25']:6.3f}  {va['pck50']:6.3f}  "
              f"{va['macro_f1']:6.3f}  {no_improve:5d}{tag}")

        if no_improve >= patience:
            print(f"\nEarly stopping: PCK@50 + F1 flat for {patience} epochs "
                  f"(best combined={best_combined:.4f})")
            break

    print(f"\nDone. Best val_loss={best_val_loss:.4f}")
    print(f"Best checkpoint : {best_ckpt_path.resolve()}")
    print(f"Last checkpoint : {(CKPT_DIR / 'last_epoch.pth').resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GCPNet")
    parser.add_argument("--epochs",     type=int,   default=40,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int,   default=16,
                        help="Batch size (reduce to 8 if memory is tight)")
    parser.add_argument("--lr",         type=float, default=1e-4,
                        help="Base LR for heads; backbone gets lr/10")
    parser.add_argument("--img-size",   type=int,   default=IMG_SIZE,
                        help="Resize target (both dims)")
    parser.add_argument("--lambda-reg", type=float, default=10.0,
                        help="Weight on regression loss relative to cls loss")
    parser.add_argument("--workers",    type=int,   default=0,
                        help="DataLoader num_workers (0 = main process, safe on Windows)")
    parser.add_argument("--resume",     type=str,   default=None,
                        help="Path to checkpoint (.pth) to resume from, e.g. checkpoints/last_epoch.pth")
    parser.add_argument("--patience",   type=int,   default=5,
                        help="Early stopping patience: epochs without PCK@50+F1 improvement")
    args = parser.parse_args()
    main(args)
