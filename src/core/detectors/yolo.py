from pathlib import Path
import numpy as np
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, weights: str, fallback_weights: str, conf: float, iou: float, device: str, class_agnostic: bool):
        w = Path(weights)
        self.model = YOLO(str(w if w.exists() else fallback_weights))
        self.conf = conf
        self.iou = iou
        self.device = device
        self.class_agnostic = class_agnostic

    def predict(self, img_bgr: np.ndarray):
        # Si class_agnostic=True, forzamos predicción sin filtrar por clase
        res = self.model.predict(
            source=img_bgr, verbose=False, conf=self.conf, iou=self.iou, device=self.device,
            classes=None if self.class_agnostic else None  # placeholder para futuro fine-tune
        )[0]
        boxes = []
        if res.boxes is not None:
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf = float(b.conf[0]) if b.conf is not None else 0.0
                cls = int(b.cls[0]) if b.cls is not None else -1
                boxes.append({"xyxy":[x1,y1,x2,y2], "conf":conf, "cls":cls})
        return boxes
