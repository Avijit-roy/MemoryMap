"""Importance scoring combining multiple signals (CCTV focused)"""

from data_structures import Scene


class ImportanceScoringEngine:
    """
    Scores scene importance for static surveillance videos
    """

    @staticmethod
    def score_scene(
        scene: Scene,
        motion_score: float,
        emotion_score: float,
        context_data: dict,
        semantic_label: str
    ) -> float:
        """
        CCTV-aware importance scoring
        """

        # 🚫 HARD FILTER: ignore idle scenes
        if semantic_label == "idle_scene" and motion_score < 0.08:
            return 0.0

        # -------------------------
        # Base signal weights
        # -------------------------
        motion_weight = motion_score * 0.55   # motion dominates CCTV
        context_weight = context_data.get("context_change", 0.0) * 0.25
        visual_weight = emotion_score * 0.05  # very minor

        # -------------------------
        # Semantic boost (CCTV)
        # -------------------------
        semantic_boost = 0.0

        if semantic_label == "new_object_appearance":
            semantic_boost = 0.30
        elif semantic_label == "significant_activity":
            semantic_boost = 0.25
        elif semantic_label == "minor_activity":
            semantic_boost = 0.10
        elif semantic_label == "background_activity":
            semantic_boost = 0.05

        # -------------------------
        # Final importance score
        # -------------------------
        total_score = (
            motion_weight
            + context_weight
            + visual_weight
            + semantic_boost
        )

        # Clamp
        return max(0.0, min(total_score, 1.0))
