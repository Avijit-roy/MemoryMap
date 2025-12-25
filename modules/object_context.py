from typing import Dict, List
import numpy as np
from modules.utils import resize_for_yolo

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from data_structures import Scene, Frame


# ✅ NEW: Object risk weighting for semantic importance
OBJECT_RISK_WEIGHTS = {
    # Critical threats
    "knife": 0.95,
    "gun": 1.0,
    "weapon": 0.98,
    "person_with_weapon": 1.0,
    
    # High importance
    "fire": 0.90,
    "smoke": 0.85,
    
    # Medium importance
    "person": 0.5,  # Depends on context (gathering vs. solo)
    "car": 0.4,
    "truck": 0.45,
    "motorcycle": 0.42,
    "bicycle": 0.3,
    
    # Low importance
    "dog": 0.2,
    "cat": 0.15,
    "backpack": 0.25,
    "phone": 0.1,
    "laptop": 0.15,
    
    # Default for unknown objects
    "default": 0.3,
}


class ObjectContextAnalyzer:
    """
    Object & context analysis optimized for static surveillance cameras
    
    Now includes semantic object class weighting to distinguish between:
    - Critical objects (knife, gun) vs. normal objects (person, car)
    - Risk-aware importance scoring instead of raw object counts
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
        """
        Analyze scene with object class weighting.
        
        Now distinguishes between:
        - Knife appearing (0.95 risk) vs Person appearing (0.5 risk)
        - Returns critical_objects list and has_weapon flag
        """
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

        # ✅ NEW: Weight by object type (semantic importance)
        weighted_importance = 0.0
        critical_objects = []
        
        for obj_class in unique_objects:
            weight = OBJECT_RISK_WEIGHTS.get(obj_class, OBJECT_RISK_WEIGHTS["default"])
            weighted_importance += weight
            
            # Track critical threats (weight >= 0.8)
            if weight >= 0.8:
                critical_objects.append(obj_class)
        
        # Average weighted importance (0-1 scale)
        context_change = (
            weighted_importance / len(unique_objects)
            if unique_objects else 0.0
        )
        
        # ✅ NEW: Detect weapons
        has_weapon = any(obj in ["knife", "gun", "weapon"] for obj in unique_objects)

        return {
            "objects": list(unique_objects),
            "object_count": len(objects),
            "context_change": min(context_change, 1.0),  # 0-1 scale
            "critical_objects": critical_objects,  # ✅ NEW: List of high-risk objects
            "has_weapon": has_weapon,  # ✅ NEW: Boolean flag for weapon detection
        }

    @staticmethod
    def _empty_result() -> Dict:
        return {
            "objects": [],
            "object_count": 0,
            "context_change": 0.0,
            "critical_objects": [],  # ✅ NEW
            "has_weapon": False,  # ✅ NEW
        }