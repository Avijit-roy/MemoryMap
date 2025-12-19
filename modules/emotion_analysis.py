import cv2
from data_structures import Scene

class EmotionAnalysis:
    """Estimates emotional intensity in scenes"""

    @staticmethod
    def estimate_emotion_score(scene: Scene) -> float:
        """Estimate emotion intensity (0-1) based on frame variance"""
        # Get representative frame
        if not scene.representative_frame:
            return 0.0  # ✅ fallback for missing frame

        frame = getattr(scene.representative_frame, "image", None)
        if frame is None:
            return 0.0  # ✅ fallback for missing image

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute Laplacian variance (focus measure)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        return min(variance / 500.0, 1.0)
