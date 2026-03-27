"""
Model comparison script — benchmarks all GazeNet architectures and ML baselines
on the same validation split and prints a ranked summary table.

Usage:
    # Compare all deep architectures (loads pre-trained weights if available)
    python3 compare_models.py --data data/MPIIGaze

    # Also run ML baselines (slower)
    python3 compare_models.py --data data/MPIIGaze --ml

    # Train each architecture from scratch before comparing
    python3 compare_models.py --data data/MPIIGaze --train --epochs 30

    # Quick test (2 epochs, small batch)
    python3 compare_models.py --data data/MPIIGaze --train --epochs 2 --batch 64
"""

import argparse
import os
import time

import torch

from src.gaze_model  import build_model, list_archs, angular_loss, angular_error
from src.dataset     import get_dataloaders
from train           import train_one_epoch, evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_arch(arch, train_loader, val_loader, device, epochs, lr, batch_size):
    model     = build_model(arch).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    best_ang   = float("inf")
    best_state = None

    print(f"  Training {arch} for {epochs} epoch(s)...")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_ang   = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_ang = evaluate(model, val_loader, device)
        scheduler.step()
        marker = " *" if val_ang < best_ang else ""
        print(f"    epoch {epoch:>3}  train {tr_ang:.2f}°  val {val_ang:.2f}°{marker}")
        if val_ang < best_ang:
            best_ang   = val_ang
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model, best_ang


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    print("Loading dataset...")
    train_loader, val_loader = get_dataloaders(
        args.data, batch_size=args.batch, num_workers=4, dataset="mpii"
    )

    results = []   # list of dicts

    # --- Deep learning architectures ---
    print("\n=== Deep Learning Architectures ===")
    for arch in list_archs():
        model_path = os.path.join("models", f"gazenet_{arch}_best.pth")

        if args.train:
            t0 = time.time()
            model, best_ang = train_arch(
                arch, train_loader, val_loader, device,
                epochs=args.epochs, lr=args.lr, batch_size=args.batch,
            )
            elapsed = time.time() - t0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), model_path)

        else:
            model = build_model(arch).to(device)
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"  Loaded weights: {model_path}")
            else:
                # Also try the legacy name for mobilenet
                legacy = "models/gazenet_best.pth"
                if arch == "mobilenet" and os.path.exists(legacy):
                    model.load_state_dict(torch.load(legacy, map_location=device))
                    print(f"  Loaded legacy weights: {legacy}")
                else:
                    print(f"  No weights found for {arch} — using random init "
                          f"(run with --train to train first)")
            t0 = time.time()
            _, best_ang = evaluate(model, val_loader, device)
            elapsed = time.time() - t0

        n_params = count_params(model)
        results.append({
            "name":     arch,
            "type":     "deep",
            "val_ang":  best_ang,
            "params":   n_params,
            "time_s":   elapsed,
        })
        print(f"  {arch:<14}  {best_ang:.2f}°  ({n_params:,} params)\n")

    # --- ML baselines ---
    if args.ml:
        print("\n=== ML Baselines ===")
        from src.ml_baselines import build_feature_matrix, get_baselines, np_angular_error
        import time as _time

        print("  Extracting features...")
        X_train, Y_train = build_feature_matrix(train_loader, max_batches=args.ml_batches)
        X_val,   Y_val   = build_feature_matrix(val_loader)
        print(f"  Train {X_train.shape[0]:,} | Val {X_val.shape[0]:,} | Features {X_train.shape[1]}")

        for name, pipeline in get_baselines().items():
            t0 = _time.time()
            pipeline.fit(X_train, Y_train)
            val_pred = pipeline.predict(X_val)
            val_ang  = np_angular_error(val_pred, Y_val)
            elapsed  = _time.time() - t0
            results.append({
                "name":    name,
                "type":    "ml",
                "val_ang": val_ang,
                "params":  None,
                "time_s":  elapsed,
            })
            print(f"  {name:<22}  {val_ang:.2f}°")

    # --- Summary table ---
    results.sort(key=lambda r: r["val_ang"])

    print("\n" + "=" * 62)
    print(f"{'Rank':<5} {'Model':<24} {'Type':<6} {'Val Ang°':>9}  {'Params':>12}")
    print("-" * 62)
    for rank, r in enumerate(results, 1):
        params_str = f"{r['params']:,}" if r["params"] else "N/A"
        print(f"{rank:<5} {r['name']:<24} {r['type']:<6} "
              f"{r['val_ang']:>9.2f}°  {params_str:>12}")
    print("=" * 62)

    best = results[0]
    print(f"\nBest model : {best['name']}  ({best['val_ang']:.2f}° angular error)")

    # Save summary CSV
    import csv
    csv_path = "models/comparison_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "type", "val_ang", "params", "time_s"])
        w.writeheader()
        w.writerows(results)
    print(f"Results saved : {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all gaze model architectures")
    parser.add_argument("--data",        default="data/MPIIGaze")
    parser.add_argument("--train",       action="store_true",
                        help="Train each architecture before evaluating")
    parser.add_argument("--epochs",      type=int,   default=30,
                        help="Epochs per architecture when --train is set")
    parser.add_argument("--batch",       type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--ml",          action="store_true",
                        help="Also benchmark ML baselines")
    parser.add_argument("--ml-batches",  type=int,   default=50,
                        help="Max train batches for ML feature extraction")
    main(parser.parse_args())
