import os
import json
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision import transforms

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ---------------------------------------------------------------------------
# Shared transforms
# ---------------------------------------------------------------------------

EYE_CROP_SIZE = (64, 32)  # (width, height)

_base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(EYE_CROP_SIZE[::-1]),   # (H, W) = (32, 64)
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

_aug_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # random cutout
])


# ---------------------------------------------------------------------------
# MPIIGaze  (original — .mat files, 15 subjects, ~427k samples)
# ---------------------------------------------------------------------------

class MPIIGazeDataset(Dataset):
    """
    Loads the MPIIGaze dataset from its native .mat format.

    Expected structure:
        <data_dir>/MPIIGaze/Data/Normalized/
            p00/  day01.mat  day02.mat  ...
            p14/

    Each .mat file contains per eye:
        image : (N, 36, 60) uint8
        gaze  : (N, 3)      float64 — 3D unit gaze vector
        pose  : (N, 3)      float64 — head pose (pitch, yaw, roll) radians

    __getitem__ returns:
        img       : (1, 32, 64) float tensor  normalised to [-1, 1]
        head_pose : (3,)        float tensor
        label     : (2,)        float tensor  — (pitch, yaw) gaze angles
    """

    def __init__(self, data_dir, augment=False):
        self.augment = augment
        self.images  = []
        self.poses   = []
        self.labels  = []
        self._load(data_dir)

    def _load(self, data_dir):
        normalized_dir = os.path.join(data_dir, "MPIIGaze", "Data", "Normalized")
        if not os.path.isdir(normalized_dir):
            raise FileNotFoundError(
                f"Could not find normalized data at: {normalized_dir}\n"
                "Expected: <data_dir>/MPIIGaze/Data/Normalized/p00/ ..."
            )

        subject_dirs = sorted([
            os.path.join(normalized_dir, d)
            for d in os.listdir(normalized_dir)
            if os.path.isdir(os.path.join(normalized_dir, d))
        ])

        n_files = 0
        for subject_dir in subject_dirs:
            for fname in sorted(os.listdir(subject_dir)):
                if fname.endswith(".mat"):
                    self._load_mat(os.path.join(subject_dir, fname))
                    n_files += 1

        self.images = np.concatenate(self.images, axis=0)
        self.poses  = np.concatenate(self.poses,  axis=0).astype(np.float32)
        self.labels = np.concatenate(self.labels, axis=0).astype(np.float32)
        print(f"[MPIIGaze]  {len(self.images):,} samples from {n_files} files "
              f"({len(subject_dirs)} subjects).")

    def _load_mat(self, path):
        mat  = sio.loadmat(path)
        data = mat["data"][0, 0]
        for side in ("left", "right"):
            try:
                eye   = data[side][0, 0]
                imgs  = eye["image"]
                gazes = eye["gaze"]
                poses = eye["pose"]
                self.images.append(imgs)
                self.poses.append(poses.astype(np.float32))
                self.labels.append(_gaze3d_to_pitchyaw(gazes))
            except (ValueError, KeyError):
                pass

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img       = self.images[idx]
        head_pose = self.poses[idx]
        label     = self.labels[idx]

        img_tensor = _base_transform(img)
        if self.augment:
            # Random horizontal flip — must negate yaw to stay consistent
            if torch.rand(1).item() < 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2])
                label = label.copy()
                label[1] = -label[1]   # flip yaw
            img_tensor = _aug_transform(img_tensor)
            # Random Gaussian noise
            img_tensor = img_tensor + torch.randn_like(img_tensor) * 0.05

        return img_tensor, torch.tensor(head_pose), torch.tensor(label)


# ---------------------------------------------------------------------------
# ETH-XGaze  (HDF5, ~110k samples, large head-pose variation)
# ---------------------------------------------------------------------------

class ETHXGazeDataset(Dataset):
    """
    Loads the ETH-XGaze dataset from HDF5 files.

    Download:
        https://ait.ethz.ch/xgaze
        Accept the licence, download train.zip / test.zip, extract to:
            <data_dir>/ETH-XGaze/train/subjectXXXX.h5

    Each HDF5 file contains:
        face_patch  : (N, 224, 224, 3) uint8  — face crop (not eye crop)
        gaze_label  : (N, 2)           float32 — (pitch, yaw) radians

    head_pose is not included in this dataset; zeros are returned for it.

    __getitem__ returns:
        img       : (1, 32, 64) float tensor
        head_pose : (3,)  zeros (ETH-XGaze doesn't provide head pose)
        label     : (2,)  (pitch, yaw) in radians
    """

    def __init__(self, data_dir, split="train", augment=False):
        if not HAS_H5PY:
            raise ImportError("h5py is required for ETHXGazeDataset. pip install h5py")
        self.augment  = augment
        self.images   = []
        self.labels   = []
        self._load(data_dir, split)

    def _load(self, data_dir, split):
        split_dir = os.path.join(data_dir, "ETH-XGaze", split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"ETH-XGaze {split} split not found at: {split_dir}\n"
                "Download from https://ait.ethz.ch/xgaze and extract to "
                "<data_dir>/ETH-XGaze/train/"
            )

        h5_files = sorted([
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith(".h5")
        ])

        for path in h5_files:
            with h5py.File(path, "r") as f:
                faces  = f["face_patch"][:]       # (N, 224, 224, 3)
                gazes  = f["gaze_label"][:]        # (N, 2)
                # Convert face patch to grayscale eye-sized crop (centre crop)
                grey = np.mean(faces, axis=-1).astype(np.uint8)   # (N, 224, 224)
                # Centre-crop to approximate eye region
                h, w = grey.shape[1], grey.shape[2]
                cy, cx = h // 2, w // 2
                ey, ex = h // 4, w // 3
                crop = grey[:, cy - ey:cy + ey, cx - ex:cx + ex]  # (N, ~112, ~149)
                self.images.append(crop)
                self.labels.append(gazes.astype(np.float32))

        if not self.images:
            raise FileNotFoundError(f"No .h5 files found in {split_dir}")

        self.images = np.concatenate(self.images, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        print(f"[ETH-XGaze] {len(self.images):,} samples ({split} split).")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img   = self.images[idx]
        label = self.labels[idx]

        img_tensor = _base_transform(img)
        if self.augment:
            img_tensor = _aug_transform(img_tensor)

        head_pose = torch.zeros(3, dtype=torch.float32)
        return img_tensor, head_pose, torch.tensor(label)


# ---------------------------------------------------------------------------
# GazeCapture  (MIT, ~2.4M samples, 1450 subjects, iPhone/iPad)
# ---------------------------------------------------------------------------

class GazeCaptureDataset(Dataset):
    """
    Loads the GazeCapture dataset (MIT).

    Download:
        https://gazecapture.csail.mit.edu/
        Sign the licence, download the tar (~130 GB), extract to:
            <data_dir>/GazeCapture/
                00002/
                    appleFace.json
                    appleLeftEye.json
                    appleRightEye.json
                    dotInfo.json
                    info.json
                    frames/
                        00001.jpg ...

    __getitem__ returns:
        img       : (1, 32, 64) float tensor  (left eye crop)
        head_pose : (3,)  zeros (not available in this dataset)
        label     : (2,)  (pitch, yaw) in radians — converted from cm dot position
    """

    def __init__(self, data_dir, augment=False, max_subjects=None):
        if not HAS_PIL:
            raise ImportError("Pillow is required for GazeCaptureDataset. pip install Pillow")
        self.augment = augment
        self.samples = []   # list of (img_path, eye_bbox, pitch, yaw)
        self._load(data_dir, max_subjects)

    def _load(self, data_dir, max_subjects):
        root = os.path.join(data_dir, "GazeCapture")
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"GazeCapture not found at: {root}\n"
                "Download from https://gazecapture.csail.mit.edu/ and extract to "
                "<data_dir>/GazeCapture/"
            )

        subjects = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and d.isdigit()
        ])
        if max_subjects:
            subjects = subjects[:max_subjects]

        loaded = 0
        for subj in subjects:
            subj_dir = os.path.join(root, subj)
            try:
                self._load_subject(subj_dir)
                loaded += 1
            except Exception:
                continue

        print(f"[GazeCapture] {len(self.samples):,} samples from {loaded} subjects.")

    def _load_subject(self, subj_dir):
        frames_dir  = os.path.join(subj_dir, "frames")
        dot_path    = os.path.join(subj_dir, "dotInfo.json")
        leye_path   = os.path.join(subj_dir, "appleLeftEye.json")
        info_path   = os.path.join(subj_dir, "info.json")

        if not all(os.path.exists(p) for p in [frames_dir, dot_path, leye_path, info_path]):
            return

        with open(dot_path)  as f: dot  = json.load(f)
        with open(leye_path) as f: leye = json.load(f)
        with open(info_path) as f: info = json.load(f)

        screen_w_cm = info.get("DeviceScreenWidthCm",  14.0)
        screen_h_cm = info.get("DeviceScreenHeightCm", 19.0)
        cam_dist_cm = 30.0  # approximate viewing distance for phone/tablet

        frames = sorted(os.listdir(frames_dir))
        for i, fname in enumerate(frames):
            if i >= len(dot["XCam"]):
                break
            if not dot["IsValid"][i]:
                continue
            # Eye bounding box (x, y, w, h) in pixels
            if not leye["IsValid"][i]:
                continue
            bbox = (
                leye["X"][i], leye["Y"][i],
                leye["W"][i], leye["H"][i],
            )
            # Dot position in cm from camera → pitch/yaw
            dx_cm = dot["XCam"][i]
            dy_cm = dot["YCam"][i]
            yaw   = float(np.arctan2(dx_cm, cam_dist_cm))
            pitch = float(np.arctan2(-dy_cm, cam_dist_cm))

            img_path = os.path.join(frames_dir, fname)
            self.samples.append((img_path, bbox, pitch, yaw))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, (bx, by, bw, bh), pitch, yaw = self.samples[idx]

        img  = Image.open(img_path).convert("L")
        bx, by, bw, bh = int(bx), int(by), int(bw), int(bh)
        if bw > 0 and bh > 0:
            img = img.crop((bx, by, bx + bw, by + bh))

        img_array  = np.array(img, dtype=np.uint8)
        img_tensor = _base_transform(img_array)
        if self.augment:
            img_tensor = _aug_transform(img_tensor)

        head_pose = torch.zeros(3, dtype=torch.float32)
        label     = torch.tensor([pitch, yaw], dtype=torch.float32)
        return img_tensor, head_pose, label


# ---------------------------------------------------------------------------
# Combined dataset
# ---------------------------------------------------------------------------

class CombinedGazeDataset(Dataset):
    """
    Merges multiple gaze datasets into one.

    Usage:
        ds = CombinedGazeDataset([
            MPIIGazeDataset("data/MPIIGaze", augment=True),
            ETHXGazeDataset("data",          augment=True),
        ])
    """

    def __init__(self, datasets):
        self._dataset = ConcatDataset(datasets)
        total = sum(len(d) for d in datasets)
        print(f"[Combined]  {total:,} samples total from {len(datasets)} datasets.")

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _gaze3d_to_pitchyaw(gaze3d):
    """Convert 3D unit gaze vectors (N,3) → (pitch, yaw) in radians (N,2)."""
    gx, gy, gz = gaze3d[:, 0], gaze3d[:, 1], gaze3d[:, 2]
    pitch = np.arcsin(-gy)
    yaw   = np.arctan2(-gx, -gz)
    return np.stack([pitch, yaw], axis=1).astype(np.float32)


def get_dataloaders(data_dir, batch_size=128, val_split=0.1, num_workers=4,
                    dataset="mpii", augment=True):
    """
    Build train/val DataLoaders.

    Args:
        data_dir   : root data directory
        batch_size : batch size
        val_split  : fraction held out for validation
        num_workers: DataLoader workers
        dataset    : 'mpii' | 'ethxgaze' | 'gazecapture' | 'combined'
        augment    : whether to use augmentation on training set

    Returns:
        (train_loader, val_loader)
    """
    if dataset == "mpii":
        full      = MPIIGazeDataset(data_dir, augment=augment)
        val_clean = MPIIGazeDataset(data_dir, augment=False)

    elif dataset == "ethxgaze":
        full      = ETHXGazeDataset(data_dir, split="train", augment=augment)
        val_clean = ETHXGazeDataset(data_dir, split="train", augment=False)

    elif dataset == "gazecapture":
        full      = GazeCaptureDataset(data_dir, augment=augment)
        val_clean = GazeCaptureDataset(data_dir, augment=False)

    elif dataset == "combined":
        ds_train = []
        ds_val   = []
        for cls, kwargs in [
            (MPIIGazeDataset,   {"data_dir": data_dir}),
            (ETHXGazeDataset,   {"data_dir": data_dir, "split": "train"}),
        ]:
            try:
                ds_train.append(cls(augment=augment,  **kwargs))
                ds_val.append(  cls(augment=False,    **kwargs))
            except FileNotFoundError as e:
                print(f"  Skipping {cls.__name__}: {e}")
        if not ds_train:
            raise RuntimeError("No datasets found for combined mode.")
        full      = CombinedGazeDataset(ds_train)
        val_clean = CombinedGazeDataset(ds_val)

    else:
        raise ValueError(f"Unknown dataset '{dataset}'. "
                         "Choose: mpii, ethxgaze, gazecapture, combined")

    n_val   = int(len(full) * val_split)
    n_train = len(full) - n_val
    train_set, val_set = random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_set = torch.utils.data.Subset(val_clean, val_set.indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
