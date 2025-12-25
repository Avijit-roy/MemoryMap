"""
Importance scoring combining multiple signals (CCTV focused)
"""

from data_structures import Scene


class ImportanceScoringEngine:
    """
    Scores scene importance for static surveillance videos
    """

    @staticmethod
    def score_scene(
        scene: Scene,
        context_data: dict,
        semantic_label: str
    ) -> dict:
        """
        Event-level importance scoring

        Returns:
            {
              "score": float (0–1),
              "level": str
            }
        """

        # -------------------------
        # Extract motion intelligence
        # -------------------------
        motion_intensity = scene.motion_mean
        motion_peak = scene.motion_peak
        suddenness = scene.suddenness
        duration = scene.duration

        # -------------------------
        # Context signals
        # -------------------------
        object_delta = min(context_data.get("context_change", 0.0), 1.0)
        object_count = context_data.get("object_count", 0)

        # -------------------------
        # Normalize duration (short events still matter)
        # -------------------------
        duration_score = min(duration / 10.0, 1.0)  # 10s ≈ full weight

        # -------------------------
        # Normalize motion (robust)
        # -------------------------
        motion_score = min(motion_intensity / 25.0, 1.0)
        suddenness_score = min(suddenness / 20.0, 1.0)

        # -------------------------
        # Core importance formula
        # -------------------------
        importance = (
            0.40 * motion_score +
            0.30 * object_delta +
            0.20 * duration_score +
            0.10 * suddenness_score
        )

        # -------------------------
        # Semantic modulation
        # -------------------------
        if semantic_label == "idle_scene":
            importance *= 0.2
        elif semantic_label == "minor_activity":
            importance *= 0.6
        elif semantic_label == "significant_activity":
            importance *= 1.1
        elif semantic_label == "critical_activity":
            importance *= 1.3

        # -------------------------
        # Penalize empty-object motion
        # -------------------------
        if object_count == 0:
            importance *= 0.5

        # -------------------------
        # Clamp
        # -------------------------
        importance = max(0.0, min(importance, 1.0))

        # -------------------------
        # Human-readable level
        # -------------------------
        if importance < 0.25:
            level = "Minor activity"
        elif importance < 0.55:
            level = "Significant disturbance"
        else:
            level = "Critical event"

        return {
            "score": importance,
            "level": level
        }
