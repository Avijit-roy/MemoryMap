import cv2
import numpy as np
from data_structures import Scene


class EmotionAnalysis:
    """
    Estimates emotional / visual intensity of a scene.
    Output range: 0.0 → 1.0
    """

    @staticmethod
    def estimate_emotion_score(scene: Scene) -> float:
        if not scene.representative_frame:
            return 0.0

        frame = getattr(scene.representative_frame, "image", None)
        if frame is None:
            return 0.0

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1️⃣ Contrast intensity (standard deviation)
            contrast = np.std(gray)

            # 2️⃣ Edge density (emotional scenes have more structure)
            edges = cv2.Canny(gray, 80, 160)
            edge_density = np.mean(edges > 0)

            # Normalize values
            contrast_score = min(contrast / 70.0, 1.0)
            edge_score = min(edge_density * 4.0, 1.0)

            # Weighted emotion score
            emotion_score = (0.6 * contrast_score) + (0.4 * edge_score)

            return float(np.clip(emotion_score, 0.0, 1.0))

        except Exception:
            # Absolute safety fallback
            return 0.0
