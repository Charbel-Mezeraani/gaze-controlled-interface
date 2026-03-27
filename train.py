"""
Training script for GazeNet variants.

Usage:
    # Train default MobileNetV3 on MPIIGaze
    python3 train.py --data data/MPIIGaze

    # Train EfficientNet with early stopping
    python3 train.py --data data/MPIIGaze --arch efficientnet --patience 5

    # Train on combined MPIIGaze + ETH-XGaze
    python3 train.py --data data --dataset combined --arch mobilenet

    # Quick smoke test
    python3 train.py --data data/MPIIGaze --epochs 2 --batch 64

Architectures : mobilenet | efficientnet | resnet18 | iris
Datasets      : mpii      | ethxgaze     | gazecapture | combined
"""

import argparse
import csv
import os
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.gaze_model import build_model, list_archs, angular_loss, angular_error
from src.dataset    import get_dataloaders


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def mixup(imgs, poses, labels, alpha=0.2):
    """Mixup augmentation — blends two samples and their labels."""
    if alpha <= 0:
        return imgs, poses, labels
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    imgs   = lam * imgs   + (1 - lam) * imgs[idx]
    poses  = lam * poses  + (1 - lam) * poses[idx]
    labels = lam * labels + (1 - lam) * labels[idx]
    return imgs, poses, labels


def train_one_epoch(model, loader, optimizer, device, mixup_alpha=0.2):
    model.train()
    total_loss = total_ang = 0.0

    for imgs, poses, labels in loader:
        imgs, poses, labels = imgs.to(device), poses.to(device), labels.to(device)
        imgs, poses, labels = mixup(imgs, poses, labels, alpha=mixup_alpha)
        optimizer.zero_grad()

        if model.__class__.__name__ == "GazeNetWithIris":
            preds = model(imgs, poses, torch.zeros(imgs.size(0), 4, device=device))
        else:
            preds = model(imgs, poses)

        loss = angular_loss(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  * imgs.size(0)
        total_ang  += angular_error(preds.detach(), labels).item() * imgs.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_ang / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = total_ang = 0.0

    for imgs, poses, labels in loader:
        imgs, poses, labels = imgs.to(device), poses.to(device), labels.to(device)

        if model.__class__.__name__ == "GazeNetWithIris":
            preds = model(imgs, poses, torch.zeros(imgs.size(0), 4, device=device))
        else:
            preds = model(imgs, poses)

        total_loss += angular_loss(preds, labels).item()  * imgs.size(0)
        total_ang  += angular_error(preds, labels).item() * imgs.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_ang / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Arch    : {args.arch}")
    print(f"Dataset : {args.dataset}")

    train_loader, val_loader = get_dataloaders(
        args.data,
        batch_size=args.batch,
        num_workers=args.workers,
        dataset=args.dataset,
    )

    model = build_model(args.arch,
                        freeze_backbone=args.freeze_backbone,
                        dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params  : {n_params:,} trainable")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"gazenet_{args.arch}_best.pth")
    log_path   = os.path.join("models", f"gazenet_{args.arch}_log.csv")

    log_file = open(log_path, "w", newline="")
    writer   = csv.writer(log_file)
    writer.writerow(["epoch", "train_loss", "train_ang",
                     "val_loss", "val_ang", "lr", "elapsed_s"])

    best_ang       = float("inf")
    patience_count = 0

    print(f"\n{'Epoch':>6}  {'Train Ang°':>10}  {'Val Ang°':>9}  {'LR':>8}")
    print("-" * 42)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_ang   = train_one_epoch(model, train_loader, optimizer, device,
                                             mixup_alpha=args.mixup)
        val_loss, val_ang = evaluate(model, val_loader, device)
        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        elapsed = time.time() - t0

        writer.writerow([epoch, f"{tr_loss:.6f}", f"{tr_ang:.4f}",
                         f"{val_loss:.6f}", f"{val_ang:.4f}",
                         f"{lr:.2e}", f"{elapsed:.1f}"])
        log_file.flush()

        improved = val_ang < best_ang
        marker   = " *" if improved else ""
        print(f"{epoch:>6}  {tr_ang:>10.2f}  {val_ang:>9.2f}  {lr:>8.2e}{marker}")

        if improved:
            best_ang       = val_ang
            patience_count = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_count += 1

        if args.patience and patience_count >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs).")
            break

    log_file.close()
    print(f"\nBest val angular error : {best_ang:.2f}°")
    print(f"Model saved            : {model_path}")
    print(f"Training log (CSV)     : {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GazeNet variant")
    parser.add_argument("--data",            default="data/MPIIGaze")
    parser.add_argument("--dataset",         default="mpii",
                        choices=["mpii", "ethxgaze", "gazecapture", "combined"])
    parser.add_argument("--arch",            default="mobilenet",
                        choices=list_archs())
    parser.add_argument("--epochs",          type=int,   default=30)
    parser.add_argument("--batch",           type=int,   default=256)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--workers",         type=int,   default=4)
    parser.add_argument("--patience",        type=int,   default=None,
                        help="Early stopping patience (epochs). None = disabled.")
    parser.add_argument("--mixup",           type=float, default=0.2,
                        help="Mixup alpha (0 = disabled, 0.2 = default)")
    parser.add_argument("--dropout",         type=float, default=0.3,
                        help="Dropout rate in regression head (default 0.3, try 0.5 for large models)")
    parser.add_argument("--weight-decay",    type=float, default=1e-4,
                        help="AdamW weight decay (default 1e-4, try 1e-3 to reduce overfitting)")
    parser.add_argument("--freeze-backbone", action="store_true")
    main(parser.parse_args())
