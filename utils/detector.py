import os
from pathlib import Path
import numpy as np
from typing import List, Dict

MODEL_NOT_LOADED = "AI MODEL NOT LOADED — DEMO MODE"

class Detector:
    def __init__(self, model_path: Path, confidence: float = 0.5):
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.model = None
        self.model_loaded = False
        # lazy import
        try:
            if self.model_path.exists():
                from ultralytics import YOLO
                self.model = YOLO(str(self.model_path))
                self.model_loaded = True
        except Exception:
            # keep model_loaded False
            self.model = None

    def detect_on_frame(self, frame: np.ndarray) -> List[Dict]:
        """Run detection on a BGR frame. Returns list of detections with normalized labels.

        Each detection dict contains: label, label_norm, conf, box (x1,y1,x2,y2)
        """
        if not self.model_loaded or self.model is None:
            return []

        # ultralytics YOLO expects RGB
        img = frame[..., ::-1]
        results = self.model(img, imgsz=640, conf=self.confidence)
        out = []
        try:
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for b in boxes:
                    conf = float(b.conf[0]) if hasattr(b, 'conf') else float(b.conf)
                    cls = b.cls[0] if hasattr(b, 'cls') else None
                    label = r.names[int(cls)] if cls is not None and r.names else str(cls)
                    # bounding box
                    xyxy = b.xyxy[0].tolist() if hasattr(b, 'xyxy') else [0,0,0,0]
                    # normalize class names: lower, replace underscores/hyphens with spaces
                    label_norm = label.strip().lower().replace('_', ' ').replace('-', ' ')
                    out.append({"label": label, "label_norm": label_norm, "conf": conf, "box": xyxy})
        except Exception:
            # fallback safe path
            pass
        return out
