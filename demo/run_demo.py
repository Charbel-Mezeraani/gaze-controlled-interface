"""
Gaze-Controlled Interface — live demo.

Usage:
    python3 demo/run_demo.py [--model models/gazenet_best.pth]
                             [--calib calibration/calib.pkl]
                             [--smoothing 0.8]
                             [--dwell]
                             [--recalibrate]
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eye_detector   import EyeDetector
from src.gaze_model     import GazeNet
from src.calibration    import GazeCalibrator
from src.cursor_control import SmoothCursorController, DwellClicker

CALIB_DEFAULT = "calibration/calib.pkl"
MODEL_DEFAULT = "models/gazenet_best.pth"


def load_model(model_path, device):
    model = GazeNet().to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'.\n"
            "Run:  python3 train.py --data data/MPIIGaze"
        )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")
    return model


def estimate_gaze(eye_result, model, device):
    """
    Run GazeNet on both eye crops with head pose, return averaged (pitch, yaw).
    """
    head_pose = torch.tensor(eye_result["head_pose"],
                             dtype=torch.float32).unsqueeze(0).to(device)
    estimates = []
    for key in ("left_crop", "right_crop"):
        crop = eye_result[key]           # (H, W, 1)
        img  = crop.squeeze(-1)          # (H, W)
        t    = torch.from_numpy(img).float() / 255.0
        t    = t.unsqueeze(0).unsqueeze(0)            # (1,1,H,W)
        t    = F.interpolate(t, size=(32, 64))
        t    = (t - 0.5) / 0.5
        with torch.no_grad():
            pred = model(t.to(device), head_pose).cpu().numpy()[0]
        estimates.append(pred)
    return np.mean(estimates, axis=0)    # (pitch, yaw)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model    = load_model(args.model, device)
    detector = EyeDetector()

    # Calibration
    calibrator = GazeCalibrator()
    os.makedirs(os.path.dirname(os.path.abspath(args.calib)), exist_ok=True)

    if not args.recalibrate and os.path.exists(args.calib):
        calibrator.load(args.calib)
    else:
        print("\nStarting calibration — follow the dot and press SPACE at each point.\n")
        calibrator.run(detector, model, device)
        calibrator.save(args.calib)

    controller = SmoothCursorController(smoothing=args.smoothing)
    dwell      = DwellClicker(dwell_secs=1.0) if args.dwell else None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("\nRunning — press Q in the debug window to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        result = detector.process(frame)
        if result is not None:
            pitch, yaw = estimate_gaze(result, model, device)
            sx, sy     = calibrator.predict(pitch, yaw)
            controller.move(sx, sy)

            if dwell:
                dwell.update(sx, sy)

            p, y_, r = result["head_pose"]
            dbg = result["annotated_frame"].copy()
            cv2.putText(dbg, f"gaze  pitch={pitch:.3f}  yaw={yaw:.3f}",
                        (10, dbg.shape[0] - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(dbg, f"head  pitch={p:.2f}  yaw={y_:.2f}  roll={r:.2f}",
                        (10, dbg.shape[0] - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
            cv2.putText(dbg, f"cursor ({sx}, {sy})",
                        (10, dbg.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        else:
            controller.reset()
            dbg = frame

        cv2.imshow("Gaze Demo — press Q to quit", dbg)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.release()
    print("Demo stopped.")


def main():
    parser = argparse.ArgumentParser(description="Gaze-Controlled Interface Demo")
    parser.add_argument("--model",       default=MODEL_DEFAULT)
    parser.add_argument("--calib",       default=CALIB_DEFAULT)
    parser.add_argument("--smoothing",   type=float, default=0.8)
    parser.add_argument("--dwell",       action="store_true")
    parser.add_argument("--recalibrate", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
