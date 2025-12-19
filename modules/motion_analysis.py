"""
Motion analysis module (PATCHED for performance & stability)
"""

import cv2
import numpy as np
from data_structures import Scene


class MotionAnalysis:
    """
    Estimates motion intensity in a scene

    PATCH:
    - Uses at most MAX_FRAMES frames
    - Uses simple frame differencing
    - Prevents CPU/RAM explosion
    """

    MAX_FRAMES = 10

    @staticmethod
    def calculate_motion_score(scene: Scene) -> float:
        frames = scene.frames

        if frames is None or len(frames) < 2:
            return 0.0

        # Limit number of frames used
        if len(frames) > MotionAnalysis.MAX_FRAMES:
            step = len(frames) // MotionAnalysis.MAX_FRAMES
            frames = frames[::step][:MotionAnalysis.MAX_FRAMES]

        motion_values = []

        for i in range(len(frames) - 1):
            f1 = frames[i].image
            f2 = frames[i + 1].image

            if f1 is None or f2 is None:
                continue

            # Convert to grayscale
            g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

            # Absolute difference
            diff = cv2.absdiff(g1, g2)
            motion = np.mean(diff)

            motion_values.append(motion)

        if not motion_values:
            return 0.0

        # Normalize motion score
        motion_score = float(np.mean(motion_values) / 255.0)
        return min(motion_score, 1.0)
