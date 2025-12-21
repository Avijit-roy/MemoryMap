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

        # 1️⃣ No activity at all
        if motion_score < 0.05 and context_change < 0.1:
            return "idle_scene"

        # 2️⃣ Strong motion (running, crowd, vehicle movement)
        if motion_score > 0.5:
            return "significant_activity"

        # 3️⃣ New object or major context change (person/car enters view)
        if context_change > 0.6:
            return "new_object_appearance"

        # 4️⃣ Small / subtle motion (walking, background movement)
        if motion_score > 0.15:
            return "minor_activity"

        # 5️⃣ Fallback
        return "background_activity"
