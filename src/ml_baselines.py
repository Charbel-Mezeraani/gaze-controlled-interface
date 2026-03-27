"""
ML baseline models for gaze estimation.

These models operate on lightweight hand-crafted features extracted directly
from the eye crop and head pose, without training a deep neural network.

Features per sample (11-dim):
    head_pose      (3)  — pitch, yaw, roll from solvePnP
    eye_mean       (1)  — mean pixel intensity of eye crop
    eye_std        (1)  — pixel std of eye crop
    pupil_cx, cy   (2)  — centroid of dark region (pupil proxy) in [0,1]
    pupil_area     (1)  — fraction of dark pixels
    h_gradient     (1)  — mean horizontal gradient (proxy for iris position)
    v_gradient     (1)  — mean vertical gradient
    iris_cx, cy    (2)  — iris centre estimate from intensity map

Usage:
    python3 src/ml_baselines.py --data data/MPIIGaze
"""

import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dataset    import MPIIGazeDataset, get_dataloaders
from src.gaze_model import angular_error


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(img_tensor, head_pose_tensor):
    """
    Extract lightweight features from a batch.

    Args:
        img_tensor       : (B, 1, H, W) float tensor in [-1, 1]
        head_pose_tensor : (B, 3)       float tensor

    Returns:
        np.ndarray (B, 11)
    """
    # Back to [0, 1] numpy
    imgs  = ((img_tensor.cpu().numpy() + 1.0) / 2.0)[:, 0]   # (B, H, W)
    poses = head_pose_tensor.cpu().numpy()                      # (B, 3)
    B, H, W = imgs.shape

    feats = np.zeros((B, 11), dtype=np.float32)
    feats[:, :3] = poses

    for i, img in enumerate(imgs):
        feats[i, 3] = img.mean()
        feats[i, 4] = img.std()

        # Dark region centroid (pupil proxy) — invert so pupil is bright
        dark  = 1.0 - img
        total = dark.sum() + 1e-8
        ys, xs = np.mgrid[0:H, 0:W]
        feats[i, 5] = (xs * dark).sum() / (total * W)   # cx in [0,1]
        feats[i, 6] = (ys * dark).sum() / (total * H)   # cy in [0,1]
        feats[i, 7] = (dark > 0.5).mean()               # dark fraction

        # Gradients
        gx = np.diff(img, axis=1)
        gy = np.diff(img, axis=0)
        feats[i, 8] = np.abs(gx).mean()
        feats[i, 9] = np.abs(gy).mean()

        # Bright-region centroid (iris proxy)
        bright = img
        btotal = bright.sum() + 1e-8
        feats[i, 10] = (xs * bright).sum() / (btotal * W)

    return feats


def build_feature_matrix(loader, max_batches=None):
    """Load a DataLoader and extract features from all batches."""
    X, Y = [], []
    for i, (imgs, poses, labels) in enumerate(loader):
        X.append(extract_features(imgs, poses))
        Y.append(labels.numpy())
        if max_batches and i + 1 >= max_batches:
            break
    return np.concatenate(X), np.concatenate(Y)


# ---------------------------------------------------------------------------
# Angular error in numpy (for sklearn models)
# ---------------------------------------------------------------------------

def np_angular_error(pred, target):
    """Mean angular error in degrees (numpy)."""
    def to_3d(gaze):
        pitch, yaw = gaze[:, 0], gaze[:, 1]
        x = -np.cos(pitch) * np.sin(yaw)
        y = -np.sin(pitch)
        z = -np.cos(pitch) * np.cos(yaw)
        return np.stack([x, y, z], axis=1)

    p3 = to_3d(pred);   p3 /= np.linalg.norm(p3, axis=1, keepdims=True) + 1e-8
    t3 = to_3d(target); t3 /= np.linalg.norm(t3, axis=1, keepdims=True) + 1e-8
    cos = np.clip((p3 * t3).sum(axis=1), -1, 1)
    return np.degrees(np.arccos(cos)).mean()


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_baselines():
    from sklearn.linear_model import Ridge
    from sklearn.svm          import SVR
    from sklearn.ensemble     import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.multioutput  import MultiOutputRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline      import Pipeline

    def make_pipeline(regressor):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model",  MultiOutputRegressor(regressor, n_jobs=-1)),
        ])

    return {
        "Ridge":             make_pipeline(Ridge(alpha=1.0)),
        "SVR (RBF)":         make_pipeline(SVR(kernel="rbf", C=10, epsilon=0.01)),
        "Random Forest":     make_pipeline(
            RandomForestRegressor(n_estimators=100, max_depth=12,
                                  random_state=42, n_jobs=-1)),
        "Gradient Boosting": make_pipeline(
            GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                      learning_rate=0.05, random_state=42)),
    }


# ---------------------------------------------------------------------------
# Main comparison runner
# ---------------------------------------------------------------------------

def run(args):
    print("Loading dataset...")
    _, val_loader = get_dataloaders(
        args.data, batch_size=512, num_workers=4, dataset="mpii"
    )
    # Use a larger portion for training baselines
    train_loader, _ = get_dataloaders(
        args.data, batch_size=512, num_workers=4, dataset="mpii"
    )

    print("Extracting features (train)...")
    X_train, Y_train = build_feature_matrix(train_loader,
                                             max_batches=args.max_batches)
    print(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")

    print("Extracting features (val)...")
    X_val, Y_val = build_feature_matrix(val_loader)
    print(f"  Val:   {X_val.shape[0]:,} samples")

    baselines = get_baselines()
    results   = {}

    print(f"\n{'Model':<22}  {'Train Ang°':>10}  {'Val Ang°':>8}  {'Time (s)':>9}")
    print("-" * 58)

    for name, pipeline in baselines.items():
        t0 = time.time()
        pipeline.fit(X_train, Y_train)
        elapsed = time.time() - t0

        train_pred = pipeline.predict(X_train)
        val_pred   = pipeline.predict(X_val)

        train_ang = np_angular_error(train_pred, Y_train)
        val_ang   = np_angular_error(val_pred,   Y_val)

        results[name] = {"train_ang": train_ang, "val_ang": val_ang}
        print(f"{name:<22}  {train_ang:>10.2f}  {val_ang:>8.2f}  {elapsed:>9.1f}s")

    best = min(results, key=lambda k: results[k]["val_ang"])
    print(f"\nBest ML baseline: {best}  ({results[best]['val_ang']:.2f}°)")
    print("\nNote: GazeNet (MobileNetV3) achieves ~4.6° — deep models significantly")
    print("      outperform hand-crafted feature baselines on this task.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML baseline comparison for gaze estimation")
    parser.add_argument("--data",        default="data/MPIIGaze")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limit training batches (faster testing)")
    run(parser.parse_args())
