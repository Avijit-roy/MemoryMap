from ultralytics import YOLO
from data_structures import Scene
import numpy as np
import cv2


class ObjectContextAnalyzer:
    """YOLOv8-based object and context analyzer"""

    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model = YOLO(model_name)

    def analyze_scene(self, scene: Scene) -> dict:
        if scene.representative_frame is None:
            return {"objects_detected": [], "context_change": 0.0}

        frame = scene.representative_frame

        results = self.model(frame, verbose=False)[0]

        objects = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            objects.append(label)

        unique_objects = list(set(objects))

        # Context change: compare first & last frame
        if len(scene.frames) >= 2:
            first = cv2.cvtColor(scene.frames[0].image, cv2.COLOR_BGR2GRAY)
            last = cv2.cvtColor(scene.frames[-1].image, cv2.COLOR_BGR2GRAY)
            change = np.mean(cv2.absdiff(first, last)) / 255.0
        else:
            change = 0.0

        return {
            "objects_detected": unique_objects,
            "context_change": min(change, 1.0)
        }
