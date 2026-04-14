"""
DeepGuard AI — Training Script
================================
Usage:
    python train.py --data_dir /path/to/dataset
    python train.py --data_dir /path/to/dataset --resume          # resume from checkpoint
    python train.py --data_dir /path/to/dataset --epochs 50 --batch_size 64
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from dataset import setup_dataloaders
from model import DeepGuardNet
from utils import LabelSmoothCE, seed_everything, setup_logging

logger = setup_logging()


# ── Data helpers ──────────────────────────────────────────────────────────────
def find_dataset(folder: str):
    """Scan `folder` recursively for directories named 'real' and 'fake'."""
    path = Path(folder)
    real_dir = fake_dir = None
    for p in sorted(path.rglob("*")):
        if not p.is_dir():
            continue
        name = p.name.lower()
        imgs = list(p.glob("*.jpg")) + list(p.glob("*.png")) + list(p.glob("*.jpeg"))
        if len(imgs) < 10:
            continue
        if "real" in name and real_dir is None:
            real_dir = p
        if any(k in name for k in ("fake", "stylegan", "synthetic")) and fake_dir is None:
            fake_dir = p
    return real_dir, fake_dir


def prepare_dataframe(real_dir: Path, fake_dir: Path) -> pd.DataFrame:
    """Build a DataFrame of image paths and labels (0=Real, 1=Fake)."""
    exts = ("*.jpg", "*.png", "*.jpeg")
    real_paths = [p for ext in exts for p in real_dir.glob(ext)]
    fake_paths = [p for ext in exts for p in fake_dir.glob(ext)]

    df_real = pd.DataFrame({"path": [str(p) for p in real_paths], "label": 0})
    df_fake = pd.DataFrame({"path": [str(p) for p in fake_paths], "label": 1})
    df = (
        pd.concat([df_real, df_fake], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    logger.info(f"Dataset  real={len(real_paths):,}  fake={len(fake_paths):,}  total={len(df):,}")
    return df


# ── Train / eval loops ────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    loss_sum = correct = total = 0

    for imgs, labs in tqdm(loader, desc="train", leave=False):
        imgs, labs = imgs.to(device, non_blocking=True), labs.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # NOTE: torch.amp.autocast replaces the deprecated torch.cuda.amp.autocast
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            out  = model(imgs)
            loss = criterion(out, labs)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        bs = imgs.size(0)
        loss_sum += loss.item() * bs
        correct  += (out.argmax(1) == labs).sum().item()
        total    += bs

    return loss_sum / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = total = 0
    probs_all, labs_all = [], []

    for imgs, labs in tqdm(loader, desc="eval", leave=False):
        imgs, labs = imgs.to(device, non_blocking=True), labs.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            out  = model(imgs)
            loss = criterion(out, labs)

        bs = imgs.size(0)
        loss_sum += loss.item() * bs
        correct  += (out.argmax(1) == labs).sum().item()
        total    += bs

        probs_all.extend(torch.softmax(out, 1)[:, 1].cpu().numpy())
        labs_all.extend(labs.cpu().numpy())

    auc = roc_auc_score(labs_all, probs_all)
    return loss_sum / total, correct / total, auc


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    ckpt_path = Path(args.checkpoint)

    # ── Dataset ──────────────────────────────────────────────────────────
    logger.info(f"Searching dataset in: {args.data_dir}")
    real_dir, fake_dir = find_dataset(args.data_dir)
    if not real_dir or not fake_dir:
        logger.error("Could not locate 'real' and 'fake' image directories.")
        return

    df = prepare_dataframe(real_dir, fake_dir)
    train_loader, val_loader, _, _ = setup_dataloaders(
        df, batch_size=args.batch_size, num_workers=args.workers
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = DeepGuardNet(backbone_name="efficientnet_b0").to(device)
    logger.info("\n" + model.param_summary())

    criterion = LabelSmoothCE(smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, eta_min=1e-6
    )
    # GradScaler: use the non-deprecated constructor
    scaler = torch.amp.GradScaler(device=device.type, enabled=(device.type == "cuda"))

    # ── Optional resume ───────────────────────────────────────────────────
    start_epoch = 1
    best_auc    = 0.0

    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_auc    = ckpt["val_auc"]
        logger.info(f"Resumed from epoch {ckpt['epoch']}  (best AUC={best_auc:.4f})")
    elif ckpt_path.exists() and not args.resume:
        logger.info(
            f"Checkpoint found at {ckpt_path}. "
            "Skipping training. Use --resume to continue or delete the file to retrain."
        )
        return

    # ── Training loop ─────────────────────────────────────────────────────
    patience_cnt  = 0
    UNFREEZE_EPOCH = 8

    for ep in range(start_epoch, args.epochs + 1):

        # Progressive unfreeze after UNFREEZE_EPOCH epochs
        if ep == UNFREEZE_EPOCH + 1:
            logger.info(f"Epoch {ep}: unfreezing full backbone for fine-tuning.")
            model.unfreeze_all()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, eta_min=1e-7
            )

        tl, ta        = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        vl, va, auc   = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        mark = " ★" if auc > best_auc else ""
        logger.info(
            f"[{ep:02d}/{args.epochs}] "
            f"train loss={tl:.4f} acc={ta:.4f} | "
            f"val loss={vl:.4f} acc={va:.4f} AUC={auc:.4f}{mark}"
        )

        if auc > best_auc:
            best_auc     = auc
            patience_cnt = 0
            torch.save(
                {
                    "epoch"      : ep,
                    "state_dict" : model.state_dict(),
                    "optimizer"  : optimizer.state_dict(),
                    "scheduler"  : scheduler.state_dict(),
                    "val_auc"    : auc,
                    "val_acc"    : va,
                    "class_names": ["Real", "Fake"],
                    "image_size" : 224,
                    "backbone"   : "efficientnet_b0",
                },
                str(ckpt_path),
            )
            logger.info(f"  → Saved checkpoint (AUC={auc:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logger.info(f"Early stopping at epoch {ep}. Best AUC: {best_auc:.4f}")
                break

    logger.info(f"Training complete. Best Val AUC: {best_auc:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepGuard AI — Trainer")
    parser.add_argument("--data_dir",   type=str,   default="/content",        help="Root dir of dataset")
    parser.add_argument("--checkpoint", type=str,   default="best_model.pth",  help="Checkpoint path")
    parser.add_argument("--resume",     action="store_true",                    help="Resume from checkpoint")
    parser.add_argument("--epochs",     type=int,   default=30,                 help="Max training epochs")
    parser.add_argument("--batch_size", type=int,   default=32,                 help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-3,               help="Initial learning rate")
    parser.add_argument("--workers",    type=int,   default=2,                  help="DataLoader workers")
    parser.add_argument("--patience",   type=int,   default=5,                  help="Early-stopping patience")
    main(parser.parse_args())
