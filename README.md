# Gaze-Controlled Interface

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS-lightgrey)
![CUDA](https://img.shields.io/badge/CUDA-optional-orange)

Control your mouse cursor using only your eyes. This project combines real-time eye tracking with a trained deep learning model to map where you are looking on the screen and move the cursor there.

> **Demo:** A calibration screen shows 16 dots one at a time — look at each and press `SPACE`. After ~2 minutes, your eyes control the cursor.

---

## How It Works

1. **Eye Detection** — MediaPipe detects your face and extracts cropped eye images and iris positions from your webcam feed in real time.
2. **Gaze Estimation** — A lightweight CNN (GazeNet) predicts pitch and yaw gaze angles from the eye crop.
3. **Calibration** — A short per-user 16-point calibration session maps predicted gaze angles to your screen coordinates using Gaussian Process Regression.
4. **Cursor Control** — The mapped gaze position drives your system cursor via PyAutoGUI with EMA smoothing.

---

## Project Structure

```
gaze-controlled-interface/
├── src/
│   ├── eye_detector.py     # MediaPipe face/iris landmark detection + eye crop extraction
│   ├── gaze_model.py       # GazeNet variants — MobileNetV3, EfficientNet-B0, ResNet18, +Iris
│   ├── dataset.py          # Dataset loaders: MPIIGaze, ETH-XGaze, GazeCapture, Combined
│   ├── ml_baselines.py     # Ridge, SVR, Random Forest, Gradient Boosting baselines
│   ├── calibration.py      # Per-user calibration: gaze → screen coordinate mapping (GP)
│   └── cursor_control.py   # Real-time cursor movement driven by gaze estimates
├── demo/
│   └── run_demo.py         # End-to-end live demo
├── models/                 # Saved model weights + training logs (CSV)
├── data/                   # Dataset files (not included — see below)
├── train.py                # Model training script (multi-arch, early stopping, CSV log)
├── compare_models.py       # Benchmark all architectures + ML baselines side-by-side
├── requirements.txt
└── setup.sh
```

---

## Setup

**Requirements:** Python 3.10+, a webcam, and optionally a CUDA GPU for training.

```bash
# 1. Clone the repo
git clone https://github.com/your-username/gaze-controlled-interface.git
cd gaze-controlled-interface

# 2. Create environment and install dependencies
bash setup.sh
source venv/bin/activate

# 3. Add yourself to the video group (Linux — required for webcam access)
sudo usermod -aG video $USER
newgrp video

# 4. Download the MediaPipe face landmarker model
wget -q -O models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

---

## Running the Demo

```bash
python3 demo/run_demo.py --model models/gazenet_mobilenet_best.pth --recalibrate
```

On first run, a **16-point calibration** screen appears — look at each dot and press `SPACE`. The calibration is saved to `calibration/calib.pkl` and reused on future runs. Always recalibrate after training a new model.

| Flag | Description |
|---|---|
| `--model PATH` | Path to trained weights (default: `models/gazenet_best.pth`) |
| `--calib PATH` | Path to calibration file (default: `calibration/calib.pkl`) |
| `--smoothing 0.8` | Cursor EMA smoothing (0–1, higher = smoother but more lag) |
| `--dwell` | Enable dwell-to-click (stare 1 second to click) |
| `--recalibrate` | Force a new calibration |

**Optimal distance:** 50–65 cm from screen. Press `Q` to quit.

---

## Datasets

Three datasets are supported. You can train on any one or all combined.

### MPIIGaze (recommended)
~427k eye images from 15 subjects. Download from the [MPIIGaze page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild) and extract to:
```
data/MPIIGaze/MPIIGaze/Data/Normalized/p00/  day01.mat ...
                                         p14/
```

### ETH-XGaze (large head-pose variation)
~110k samples. Requires signing a licence at [ait.ethz.ch/xgaze](https://ait.ethz.ch/xgaze). Extract to:
```
data/ETH-XGaze/train/subject0000.h5 ...
```

### GazeCapture (most diverse — 1450 subjects)
~2.4M images from iPhone/iPad. Download from [gazecapture.csail.mit.edu](https://gazecapture.csail.mit.edu/). Extract to:
```
data/GazeCapture/00002/ 00003/ ...
```

---

## Training

### Best configuration (recommended)
```bash
python3 train.py --data data/MPIIGaze --arch mobilenet \
  --lr 1e-3 --epochs 40 --patience 8 \
  --mixup 0.2 --dropout 0.4
```

### Other architectures
```bash
# EfficientNet-B0
python3 train.py --data data/MPIIGaze --arch efficientnet \
  --lr 2e-4 --epochs 40 --patience 10 --mixup 0.3 --dropout 0.5 --weight-decay 1e-3

# ResNet-18
python3 train.py --data data/MPIIGaze --arch resnet18 --patience 8 --epochs 30

# MobileNetV3 + iris position features
python3 train.py --data data/MPIIGaze --arch iris --patience 8 --epochs 30

# Combined dataset
python3 train.py --data data --dataset combined --arch mobilenet
```

### All training arguments

| Argument | Default | Description |
|---|---|---|
| `--data` | `data/MPIIGaze` | Root data directory |
| `--dataset` | `mpii` | `mpii` / `ethxgaze` / `gazecapture` / `combined` |
| `--arch` | `mobilenet` | `mobilenet` / `efficientnet` / `resnet18` / `iris` |
| `--epochs` | `30` | Max training epochs |
| `--batch` | `256` | Batch size |
| `--lr` | `1e-3` | Learning rate |
| `--patience` | `None` | Early stopping patience (epochs) |
| `--mixup` | `0.2` | Mixup alpha — blends two samples to reduce overfitting (0 = off) |
| `--dropout` | `0.3` | Dropout rate in regression head |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--freeze-backbone` | off | Freeze backbone weights (faster, lower accuracy) |

Each run saves:
- `models/gazenet_<arch>_best.pth` — best checkpoint (lowest val angular error)
- `models/gazenet_<arch>_log.csv` — per-epoch training log

### Data augmentation

Training applies the following augmentations to reduce overfitting:

| Augmentation | Effect |
|---|---|
| Horizontal flip + yaw negation | Doubles effective training samples |
| ColorJitter (brightness, contrast) | Lighting robustness |
| Gaussian blur | Sharpness invariance |
| Random erasing (cutout) | Forces model to use whole eye region |
| Gaussian noise | Pixel-level robustness |
| Mixup | Softens decision boundaries — best regularizer for small datasets |

---

## Model Comparison

Benchmark all architectures and ML baselines on the same validation split:

```bash
# Compare using existing saved weights
python3 compare_models.py --data data/MPIIGaze

# Train all architectures from scratch, then compare
python3 compare_models.py --data data/MPIIGaze --train --epochs 30

# Include ML baselines
python3 compare_models.py --data data/MPIIGaze --ml
```

Results are saved to `models/comparison_results.csv`.

---

## ML Baselines

```bash
python3 -m src.ml_baselines --data data/MPIIGaze
```

Features: head pose (3) + eye intensity statistics + pupil centroid estimate (11 total).
Models: Ridge Regression, SVR (RBF), Random Forest, Gradient Boosting.

---

## Results

Trained on MPIIGaze (427k samples, 15 subjects) with augmentation + Mixup.

| Architecture | Val Angular Error | Params | Notes |
|---|---|---|---|
| **MobileNetV3-Small** | **4.05°** | **1.1M** | **Best — recommended** |
| MobileNetV3 + Iris | ~4.2° | 1.1M | Adds iris position features |
| ResNet-18 | ~4.5° | 11.3M | Larger, similar accuracy |
| EfficientNet-B0 | ~5–7° | 4.4M | Overfits on 15 subjects |
| Gradient Boosting (ML) | ~9–10° | — | No CNN features |
| Ridge Regression (ML) | ~9–10° | — | Linear baseline |

In screen terms at 60 cm, **4.05° ≈ 4.3 cm** of cursor error — corrected by per-user calibration.

---

## Model Architectures

All models share the same regression head and angular loss:

| Architecture | Backbone features | Extra inputs | Notes |
|---|---|---|---|
| `mobilenet` | MobileNetV3-Small (576) | head_pose (3) | Fastest, best on this dataset |
| `efficientnet` | EfficientNet-B0 (1280) | head_pose (3) | Needs more data to shine |
| `resnet18` | ResNet-18 (512) | head_pose (3) | Classic baseline |
| `iris` | MobileNetV3-Small (576) | head_pose (3) + iris_ratio (4) | Uses iris position features |

Loss: **angular loss** — minimises the angle between predicted and ground-truth gaze vectors in 3D space.

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Webcam capture, image processing, head pose (solvePnP) |
| `mediapipe` | Face & iris landmark detection |
| `torch` / `torchvision` | GazeNet training and inference |
| `numpy` / `scipy` | Numerical computation + `.mat` file loading |
| `h5py` | ETH-XGaze HDF5 loading |
| `Pillow` | GazeCapture JPEG loading |
| `pyautogui` | Cursor control |
| `scikit-learn` | GP calibration + ML baselines |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Camera FAIL` on Linux | `sudo usermod -aG video $USER` then log out/in |
| Cursor stays in centre | Redo calibration: `--recalibrate` |
| Face not detected | Improve lighting, sit 50–65 cm from screen |
| Cursor too jittery | Increase smoothing: `--smoothing 0.9` |
| Cursor too laggy | Decrease smoothing: `--smoothing 0.6` |
| High angular error after training | Use `--mixup 0.2 --dropout 0.4` to reduce overfitting |

---

## License

MIT
