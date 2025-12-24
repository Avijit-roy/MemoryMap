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
        motion_score: float,
        saliency_score: float,   # renamed (emotion ≠ CCTV)
        context_data: dict,
        semantic_label: str
    ) -> float:
        """
        CCTV-aware importance scoring
        """

        # -------------------------
        # HARD FILTER (idle noise)
        # -------------------------
        if semantic_label == "idle_scene" and motion_score < 0.06:
            return 0.0

        # -------------------------
        # Normalize context signals
        # -------------------------
        context_change = min(context_data.get("context_change", 0.0), 1.0)
        object_count = context_data.get("object_count", 0)

        # -------------------------
        # Base signal weights
        # -------------------------
        motion_weight = motion_score * 0.55        # primary
        context_weight = context_change * 0.25     # object/scene change
        visual_weight = saliency_score * 0.05      # very minor

        # -------------------------
        # Semantic boost (context-aware)
        # -------------------------
        semantic_boost = 0.0

        if semantic_label == "new_object_appearance":
            semantic_boost = 0.30 if motion_score > 0.1 else 0.15
        elif semantic_label == "significant_activity":
            semantic_boost = 0.25
        elif semantic_label == "minor_activity":
            semantic_boost = 0.10
        elif semantic_label == "background_activity":
            semantic_boost = 0.05

        # -------------------------
        # Combine signals
        # -------------------------
        total_score = (
            motion_weight
            + context_weight
            + visual_weight
            + semantic_boost
        )

        # -------------------------
        # Penalize empty-object scenes
        # -------------------------
        if object_count == 0:
            total_score *= 0.6

        # -------------------------
        # Final clamp
        # -------------------------
        return max(0.0, min(total_score, 1.0))
