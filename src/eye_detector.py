import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

MODEL_PATH = "models/face_landmarker.task"

# FaceMesh iris landmark indices
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

LEFT_EYE_CORNERS  = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]

LEFT_EYE_REGION  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_REGION = [362, 385, 387, 263, 373, 380]

# Canonical 3D face model points for head pose estimation (MediaPipe indices)
# Nose tip, chin, left eye corner, right eye corner, left mouth, right mouth
_POSE_POINT_IDS = [1, 152, 33, 263, 61, 291]
_POSE_MODEL_3D  = np.array([
    [0.0,    0.0,    0.0   ],   # Nose tip
    [0.0,   -330.0, -65.0  ],   # Chin
    [-225.0,  170.0, -135.0],   # Left eye corner
    [225.0,   170.0, -135.0],   # Right eye corner
    [-150.0, -150.0, -125.0],   # Left mouth
    [150.0,  -150.0, -125.0],   # Right mouth
], dtype=np.float64)


class EyeDetector:
    def __init__(self, eye_crop_size=(64, 32), model_path=MODEL_PATH):
        self.eye_crop_size = eye_crop_size

        options = FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.detector = FaceLandmarker.create_from_options(options)

    def process(self, frame):
        """
        Process a BGR frame and return extracted eye + head pose data.

        Returns a dict with:
            left_crop, right_crop     : (64x32) grayscale eye images as (H,W,1)
            left_iris_ratio,
            right_iris_ratio          : (x, y) iris position ratio in [0,1]
            head_pose                 : (3,) float32 — (pitch, yaw, roll) in radians
            landmarks                 : (478, 2) pixel coords
            annotated_frame           : frame with drawn landmarks
        Returns None if no face is detected.
        """
        if frame is None:
            return None
        h, w = frame.shape[:2]
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.detector.detect(mp_image)
        if not result.face_landmarks:
            return None

        lm  = result.face_landmarks[0]
        pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)

        annotated   = frame.copy()
        left_crop,  left_ratio  = self._extract_eye(frame, pts, LEFT_EYE_CORNERS,  LEFT_IRIS)
        right_crop, right_ratio = self._extract_eye(frame, pts, RIGHT_EYE_CORNERS, RIGHT_IRIS)
        head_pose               = self._estimate_head_pose(pts, w, h)
        self._draw_landmarks(annotated, pts)

        return {
            "left_crop":        left_crop,
            "right_crop":       right_crop,
            "left_iris_ratio":  left_ratio,
            "right_iris_ratio": right_ratio,
            "head_pose":        head_pose,
            "landmarks":        pts,
            "annotated_frame":  annotated,
        }

    def _estimate_head_pose(self, pts, w, h):
        """
        Estimate head pose (pitch, yaw, roll) in radians via solvePnP.
        Returns a (3,) float32 array.
        """
        image_points = pts[_POSE_POINT_IDS].astype(np.float64)

        focal   = w  # rough approximation
        cam_mat = np.array([
            [focal, 0,     w / 2],
            [0,     focal, h / 2],
            [0,     0,     1    ],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        ok, rvec, _ = cv2.solvePnP(
            _POSE_MODEL_3D, image_points, cam_mat, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return np.zeros(3, dtype=np.float32)

        # Rotation vector -> rotation matrix -> Euler angles
        rot_mat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2( rot_mat[2, 1], rot_mat[2, 2])
            yaw   = np.arctan2(-rot_mat[2, 0], sy)
            roll  = np.arctan2( rot_mat[1, 0], rot_mat[0, 0])
        else:
            pitch = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
            yaw   = np.arctan2(-rot_mat[2, 0], sy)
            roll  = 0.0

        return np.array([pitch, yaw, roll], dtype=np.float32)

    def _extract_eye(self, frame, pts, corner_ids, iris_ids):
        h, w = frame.shape[:2]
        cx1, cx2 = pts[corner_ids[0]], pts[corner_ids[1]]

        eye_w  = int(np.linalg.norm(cx2 - cx1) * 2.0)
        eye_h  = eye_w // 2
        center = ((cx1 + cx2) / 2).astype(int)

        x1 = max(center[0] - eye_w // 2, 0)
        y1 = max(center[1] - eye_h // 2, 0)
        x2 = min(x1 + eye_w, w)
        y2 = min(y1 + eye_h, h)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            blank = np.zeros((self.eye_crop_size[1], self.eye_crop_size[0], 1), dtype=np.uint8)
            return blank, (0.5, 0.5)

        crop_gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_resized = cv2.resize(crop_gray, self.eye_crop_size)
        crop_resized = crop_resized[..., np.newaxis]

        iris_center = pts[iris_ids].mean(axis=0)
        ratio_x = float(np.clip((iris_center[0] - x1) / max(x2 - x1, 1), 0, 1))
        ratio_y = float(np.clip((iris_center[1] - y1) / max(y2 - y1, 1), 0, 1))

        return crop_resized, (ratio_x, ratio_y)

    def _draw_landmarks(self, frame, pts):
        for idx in LEFT_IRIS + RIGHT_IRIS:
            cv2.circle(frame, tuple(pts[idx].astype(int)), 2, (0, 255, 0), -1)
        for idx in LEFT_EYE_REGION + RIGHT_EYE_REGION:
            cv2.circle(frame, tuple(pts[idx].astype(int)), 1, (255, 200, 0), -1)

    def release(self):
        self.detector.close()


def test_webcam():
    """Quick visual test — press Q to quit."""
    detector = EyeDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Webcam open. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        result = detector.process(frame)

        if result:
            annotated = result["annotated_frame"]

            for side, key in [("L", "left_crop"), ("R", "right_crop")]:
                crop     = result[key]
                crop_vis = cv2.resize(crop, (128, 64))
                crop_vis = cv2.cvtColor(crop_vis, cv2.COLOR_GRAY2BGR)
                x        = 10 if side == "L" else 150
                annotated[10:74, x:x+128] = crop_vis

            lx, ly = result["left_iris_ratio"]
            rx, ry = result["right_iris_ratio"]
            p, y_, r = result["head_pose"]
            cv2.putText(annotated, f"L iris: ({lx:.2f}, {ly:.2f})", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(annotated, f"R iris: ({rx:.2f}, {ry:.2f})", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(annotated, f"Pose  pitch={p:.2f} yaw={y_:.2f} roll={r:.2f}",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

            cv2.imshow("Eye Detector", annotated)
        else:
            cv2.imshow("Eye Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.release()


if __name__ == "__main__":
    test_webcam()
