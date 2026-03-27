import time
import numpy as np
import pyautogui

# Disable pyautogui fail-safe pause (we handle safety ourselves)
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True   # moving mouse to corner still raises exception


class SmoothCursorController:
    """
    Moves the system cursor to gaze-predicted screen positions with
    exponential smoothing to reduce jitter.

    Usage:
        ctrl = SmoothCursorController(smoothing=0.8)
        ctrl.move(x, y)      # call each frame
        ctrl.reset()         # clear smoothing state (e.g. after blink)
    """

    def __init__(self, smoothing=0.8, min_move_px=2):
        """
        Args:
            smoothing    : EMA coefficient in [0, 1).
                           Higher = smoother but more lag.
            min_move_px  : Ignore movements smaller than this (sub-pixel jitter).
        """
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")

        self.smoothing    = smoothing
        self.min_move_px  = min_move_px
        self._sx          = None   # smoothed x
        self._sy          = None   # smoothed y
        self.screen_w, self.screen_h = pyautogui.size()

    def move(self, x, y):
        """
        Update smoothed position and move the cursor.

        Args:
            x, y : raw screen pixel coordinates from the calibrator
        """
        x = int(np.clip(x, 0, self.screen_w - 1))
        y = int(np.clip(y, 0, self.screen_h - 1))

        if self._sx is None:
            # First call — jump directly
            self._sx, self._sy = float(x), float(y)
        else:
            a = self.smoothing
            self._sx = a * self._sx + (1 - a) * x
            self._sy = a * self._sy + (1 - a) * y

        tx, ty = int(self._sx), int(self._sy)

        cx, cy = pyautogui.position()
        if abs(tx - cx) < self.min_move_px and abs(ty - cy) < self.min_move_px:
            return

        try:
            pyautogui.moveTo(tx, ty)
        except pyautogui.FailSafeException:
            pass

    def reset(self):
        """Clear smoothing state — next call will jump directly to new position."""
        self._sx = None
        self._sy = None

    @property
    def position(self):
        """Last smoothed position, or None before first call."""
        if self._sx is None:
            return None
        return int(self._sx), int(self._sy)


class DwellClicker:
    """
    Fires a mouse click when the gaze stays within a radius for a set duration.

    Usage:
        dwell = DwellClicker(dwell_secs=1.0, radius_px=40)
        dwell.update(x, y)   # call each frame — clicks automatically
    """

    def __init__(self, dwell_secs=1.0, radius_px=40, cooldown_secs=1.5):
        self.dwell_secs   = dwell_secs
        self.radius_px    = radius_px
        self.cooldown_secs = cooldown_secs
        self._anchor_x    = None
        self._anchor_y    = None
        self._dwell_start = None
        self._last_click  = 0.0

    def update(self, x, y):
        """
        Feed the current gaze/cursor position each frame.
        Automatically calls pyautogui.click() when dwell threshold is met.

        Returns True if a click was fired this frame.
        """
        now = time.time()

        # Cooldown — avoid double-clicking
        if now - self._last_click < self.cooldown_secs:
            return False

        if self._anchor_x is None:
            self._anchor_x = x
            self._anchor_y = y
            self._dwell_start = now
            return False

        dist = np.hypot(x - self._anchor_x, y - self._anchor_y)

        if dist > self.radius_px:
            # Gaze moved — reset anchor
            self._anchor_x = x
            self._anchor_y = y
            self._dwell_start = now
            return False

        if now - self._dwell_start >= self.dwell_secs:
            pyautogui.click(x, y)
            self._last_click = now
            self._anchor_x   = None
            return True

        return False
