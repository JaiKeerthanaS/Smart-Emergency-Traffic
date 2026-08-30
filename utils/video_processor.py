import cv2
from pathlib import Path
import numpy as np


class VideoProcessor:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.cap = None
        self.available = False
        self._open()

    def _open(self):
        try:
            if not self.path.exists():
                self.available = False
                return
            self.cap = cv2.VideoCapture(str(self.path))
            self.available = self.cap.isOpened()
        except Exception:
            self.available = False

    def read_frame(self):
        """Read next frame (BGR). Loops video when it ends. Returns None if unavailable."""
        if not self.available or self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            # try to loop
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    return None
            except Exception:
                return None
        # optionally resize to reasonable size for dashboard
        try:
            h, w = frame.shape[:2]
            maxw = 640
            if w > maxw:
                scale = maxw / w
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        except Exception:
            pass
        return frame

    def peek_frame(self):
        """Return a frame for inference without advancing the internal counter if possible."""
        return self.read_frame()

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
