from typing import Dict, List
import numpy as np
from modules.utils import resize_for_yolo

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from data_structures import Scene, Frame


class ObjectContextAnalyzer:
    """
    Object & context analysis optimized for static surveillance cameras
    """

    IMPORTANT_OBJECTS = {"person", "car", "truck", "bus", "motorcycle", "bicycle"}

    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.4):
        self.conf = conf
        self.model = None

        if YOLO is not None:
            try:
                self.model = YOLO(model_name)
                print("✓ YOLO model loaded for object analysis")
            except Exception as e:
                print("⚠️ YOLO load failed:", e)

    def analyze_scene(self, scene: Scene) -> Dict:
        # Safety fallback
        if self.model is None or scene.representative_frame is None:
            return self._empty_result()

        # 🔹 Extract image safely
        rf = scene.representative_frame
        if isinstance(rf, Frame):
            frame = rf.image
        elif isinstance(rf, np.ndarray):
            frame = rf
        else:
            return self._empty_result()

        # Validate frame
        if frame is None or frame.ndim != 3:
            return self._empty_result()

        resized = resize_for_yolo(frame)

        # YOLO inference (ONCE)
        try:
            results = self.model(
                resized,
                conf=self.conf,
                verbose=False
            )[0]
        except Exception as e:
            print("⚠️ YOLO inference failed:", e)
            return self._empty_result()

        objects: List[str] = []

        if results.boxes is not None:
            for cls_id in results.boxes.cls.cpu().numpy():
                label = self.model.names.get(int(cls_id), "unknown")
                objects.append(label)

        unique_objects = set(objects)

        # 🔹 CCTV-aware context change
        important_hits = unique_objects.intersection(self.IMPORTANT_OBJECTS)

        context_change = min(
            (len(unique_objects) * 0.2) + (len(important_hits) * 0.4),
            1.0
        )

        return {
            "objects": list(unique_objects),
            "object_count": len(objects),
            "context_change": round(context_change, 3),
        }

    @staticmethod
    def _empty_result() -> Dict:
        return {
            "objects": [],
            "object_count": 0,
            "context_change": 0.0,
        }
