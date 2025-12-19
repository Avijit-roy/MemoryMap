"""Object and context analysis using YOLO (patched for memory safety)"""

from typing import Dict, List
from modules.utils import resize_for_yolo

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from data_structures import Scene


class ObjectContextAnalyzer:
    """
    Analyzes object presence and context change in a scene.

    PATCH:
    - Runs YOLO only ONCE per scene (representative frame)
    - Prevents RAM/GPU crash
    """

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
        """
        Analyze a scene using ONLY the representative frame.

        Args:
            scene: Scene object with representative_frame set

        Returns:
            Dictionary with detected objects, object count, and context change score
        """

        # Safety fallback
        if self.model is None or scene.representative_frame is None:
            return {
                "objects": [],
                "object_count": 0,
                "context_change": 0.0,
            }

        frame = scene.representative_frame.image
        resized = resize_for_yolo(frame)  # ✅ fixed order

        # Run YOLO ONCE
        try:
            results = self.model(
                resized,
                conf=self.conf,
                verbose=False
            )[0]
        except Exception as e:
            print("⚠️ YOLO inference failed:", e)
            return {
                "objects": [],
                "object_count": 0,
                "context_change": 0.0,
            }

        objects: List[str] = []

        if results.boxes is not None:
            for cls_id in results.boxes.cls.cpu().numpy():
                label = self.model.names.get(int(cls_id), "unknown")
                objects.append(label)

        # Context change heuristic (safe + cheap)
        unique_objects = set(objects)
        context_change = min(len(unique_objects) / 10.0, 1.0)

        return {
            "objects": list(unique_objects),
            "object_count": len(objects),
            "context_change": context_change,
        }
