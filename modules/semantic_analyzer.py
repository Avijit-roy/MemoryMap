"""Semantic understanding and scene classification (CCTV focused)"""

from data_structures import Scene


class SemanticAnalyzer:
    """
    Classifies motion events semantically for static surveillance videos
    """

    @staticmethod
    def classify_scene(
        scene: Scene,
        motion_score: float,
        context_change: float
    ) -> str:
        """
        CCTV-aware semantic classification
        """

        # No activity at all
        if motion_score < 0.05 and context_change < 0.1:
            return "idle_scene"

        # New object or environment change
        if context_change > 0.6:
            return "new_object_appearance"

        # Strong motion (running, crowd, vehicle)
        if motion_score > 0.5:
            return "significant_activity"

        # Small motion (walking, subtle movement)
        if motion_score > 0.15:
            return "minor_activity"

        # Fallback
        return "background_activity"
