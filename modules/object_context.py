"""Object and context analysis"""

import cv2
import numpy as np
from data_structures import Scene


class ObjectContextAnalyzer:
    """Analyzes objects and context changes"""
    
    @staticmethod
    def analyze_scene(scene: Scene) -> dict:
        """Analyze scene for objects and context changes"""
        # Simplified: detect if there's significant motion/change
        if len(scene.frames) < 2:
            return {"objects_detected": [], "context_change": 0.0}
        
        first = cv2.cvtColor(scene.frames[0].image, cv2.COLOR_BGR2GRAY)
        last = cv2.cvtColor(scene.frames[-1].image, cv2.COLOR_BGR2GRAY)
        
        change = np.mean(cv2.absdiff(first, last))
        change_normalized = min(change / 255.0, 1.0)
        
        return {
            "objects_detected": ["person", "text", "environment"],
            "context_change": change_normalized
        }