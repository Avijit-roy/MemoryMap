"""Estimates emotional/visual intensity of a scene"""

import cv2
import numpy as np
from data_structures import Scene

# Constants for emotion analysis
CONTRAST_WEIGHT = 0.6
EDGE_WEIGHT = 0.4
CONTRAST_NORM = 70.0
EDGE_DENSITY_SCALE = 4.0
CANNY_LOW = 80
CANNY_HIGH = 160


class EmotionAnalysis:
    """Estimates emotional/visual intensity of a scene. Output range: 0.0 → 1.0"""

    @staticmethod
    def estimate_emotion_score(scene: Scene) -> float:
        """
        Calculate emotion/intensity score from scene
        
        Args:
            scene: Scene object with representative_frame
            
        Returns:
            float: Emotion score 0.0-1.0
        """
        if not scene.representative_frame:
            return 0.0

        frame = None
        if hasattr(scene.representative_frame, 'image'):
            frame = scene.representative_frame.image
        elif isinstance(scene.representative_frame, np.ndarray):
            frame = scene.representative_frame
        
        if frame is None or frame.ndim != 3:
            return 0.0

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1. Contrast intensity (standard deviation)
            contrast = np.std(gray)

            # 2. Edge density (emotional scenes have more structure)
            edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
            edge_density = np.mean(edges > 0)

            # Normalize values
            contrast_score = min(contrast / CONTRAST_NORM, 1.0)
            edge_score = min(edge_density * EDGE_DENSITY_SCALE, 1.0)

            # Weighted emotion score
            emotion_score = (CONTRAST_WEIGHT * contrast_score) + (EDGE_WEIGHT * edge_score)
            return float(np.clip(emotion_score, 0.0, 1.0))

        except Exception as e:
            print(f"⚠️ Emotion analysis failed: {e}")
            return 0.0

