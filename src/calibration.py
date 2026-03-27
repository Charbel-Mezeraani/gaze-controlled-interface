import os
import time
import pickle
import numpy as np
import cv2
import pyautogui
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


# 16-point calibration grid (normalised screen coords 0..1)
CALIBRATION_POINTS = [
    (0.1, 0.1), (0.4, 0.1), (0.6, 0.1), (0.9, 0.1),
    (0.1, 0.4), (0.4, 0.4), (0.6, 0.4), (0.9, 0.4),
    (0.1, 0.6), (0.4, 0.6), (0.6, 0.6), (0.9, 0.6),
    (0.1, 0.9), (0.4, 0.9), (0.6, 0.9), (0.9, 0.9),
]


class GazeCalibrator:
    """
    Maps raw (pitch, yaw) gaze predictions to screen pixel coordinates
    using Gaussian Process Regression for accurate, smooth interpolation.

    Workflow:
        1. Show a dot at each of 16 calibration points.
        2. Collect gaze samples per point while the user fixates.
        3. Fit a GP regressor: gaze angles → screen coords.
        4. Call predict(pitch, yaw) to get (x, y) screen pixels.
    """

    def __init__(self, samples_per_point=40, dot_radius=18, warmup_secs=0.8):
        self.samples_per_point = samples_per_point
        self.dot_radius        = dot_radius
        self.warmup_secs       = warmup_secs
        self._gp_x             = None
        self._gp_y             = None
        self.screen_w, self.screen_h = pyautogui.size()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, eye_detector, gaze_model, device):
        """
        Interactive calibration: display dots, collect gaze, fit GP model.

        Args:
            eye_detector : EyeDetector instance
            gaze_model   : GazeNet instance (eval mode, on device)
            device       : torch.device
        """
        import torch

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam for calibration.")

        gaze_inputs    = []
        screen_targets = []

        win = "Calibration — look at the dot and press SPACE"
        cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        total = len(CALIBRATION_POINTS)
        for i, (norm_x, norm_y) in enumerate(CALIBRATION_POINTS):
            canvas = self._make_canvas(norm_x, norm_y, i, total, collecting=False)
            cv2.imshow(win, canvas)
            cv2.waitKey(1)

            while True:
                key = cv2.waitKey(30) & 0xFF
                if key == ord(' '):
                    break
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    raise RuntimeError("Calibration cancelled.")

            # Warmup
            t0 = time.time()
            while time.time() - t0 < self.warmup_secs:
                cap.read()

            # Collect
            collected = 0
            while collected < self.samples_per_point:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)

                result = eye_detector.process(frame)
                if result is None:
                    continue

                pitch, yaw = self._estimate_gaze(result, gaze_model, device)
                if pitch is None:
                    continue

                gaze_inputs.append([pitch, yaw])
                screen_targets.append([norm_x, norm_y])
                collected += 1

                progress = collected / self.samples_per_point
                canvas   = self._make_canvas(norm_x, norm_y, i, total,
                                             collecting=True, progress=progress)
                cv2.imshow(win, canvas)
                cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()

        self._fit(np.array(gaze_inputs), np.array(screen_targets))
        print("Calibration complete.")

    def predict(self, pitch, yaw):
        """
        Map (pitch, yaw) → (x, y) screen pixel coordinates.
        Returns (int, int) clamped to screen bounds.
        """
        if self._gp_x is None:
            raise RuntimeError("Calibrator not fitted. Call .run() first.")

        feat = np.array([[pitch, yaw]])
        nx   = float(np.clip(self._gp_x.predict(feat)[0], 0.0, 1.0))
        ny   = float(np.clip(self._gp_y.predict(feat)[0], 0.0, 1.0))
        return int(nx * self.screen_w), int(ny * self.screen_h)

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"gp_x": self._gp_x, "gp_y": self._gp_y}, f)
        print(f"Calibration saved to {path}")

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._gp_x = data["gp_x"]
        self._gp_y = data["gp_y"]
        print(f"Calibration loaded from {path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit(self, gaze_inputs, screen_targets):
        """Fit two independent GP regressors: gaze → screen_x and screen_y."""
        kernel = RBF(length_scale=0.1, length_scale_bounds=(1e-3, 10.0)) \
               + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))

        self._gp_x = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                               normalize_y=True)
        self._gp_y = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                               normalize_y=True)
        self._gp_x.fit(gaze_inputs, screen_targets[:, 0])
        self._gp_y.fit(gaze_inputs, screen_targets[:, 1])

    def _estimate_gaze(self, eye_result, gaze_model, device):
        """Run GazeNet on both eyes with head pose and return averaged (pitch, yaw)."""
        import torch
        import torch.nn.functional as F

        head_pose = torch.tensor(eye_result["head_pose"],
                                 dtype=torch.float32).unsqueeze(0).to(device)
        estimates = []
        for key in ("left_crop", "right_crop"):
            crop = eye_result[key]
            img  = crop.squeeze(-1)
            t    = torch.from_numpy(img).float() / 255.0
            t    = t.unsqueeze(0).unsqueeze(0)
            t    = F.interpolate(t, size=(32, 64))
            t    = (t - 0.5) / 0.5
            with torch.no_grad():
                pred = gaze_model(t.to(device), head_pose).cpu().numpy()[0]
            estimates.append(pred)

        if not estimates:
            return None, None

        avg = np.mean(estimates, axis=0)
        return float(avg[0]), float(avg[1])

    def _make_canvas(self, norm_x, norm_y, idx, total, collecting, progress=0.0):
        h, w   = self.screen_h, self.screen_w
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        cx, cy = int(norm_x * w), int(norm_y * h)
        r      = self.dot_radius

        cv2.circle(canvas, (cx, cy), r, (60, 60, 60), 2)
        if collecting:
            cv2.circle(canvas, (cx, cy), max(int(r * progress), 2), (0, 220, 100), -1)
        else:
            cv2.circle(canvas, (cx, cy), 6, (200, 200, 200), -1)

        msg = f"Point {idx+1}/{total} — look at the dot, press SPACE to collect" \
              if not collecting else f"Collecting... {int(progress*100)}%"
        cv2.putText(canvas, msg, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 1)
        return canvas
