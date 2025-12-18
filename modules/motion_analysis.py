"""Motion analysis in scenes"""

import cv2
import numpy as np
from data_structures import Scene


class MotionAnalysis:
    """Measures motion intensity in scenes"""
    
    @staticmethod
    def calculate_motion_score(scene: Scene) -> float:
        """Calculate motion score for scene (0-1)"""
        if len(scene.frames) < 2:
            return 0.0
        
        motion_scores = []
        
        for i in range(1, len(scene.frames)):
            prev = cv2.cvtColor(scene.frames[i-1].image, cv2.COLOR_BGR2GRAY)
            curr = cv2.cvtColor(scene.frames[i].image, cv2.COLOR_BGR2GRAY)
            
            # Optical flow approximation
            diff = np.mean(cv2.absdiff(prev, curr))
            motion_scores.append(diff)
        
        avg_motion = np.mean(motion_scores) if motion_scores else 0.0
        return min(avg_motion / 100.0, 1.0)  # Normalize