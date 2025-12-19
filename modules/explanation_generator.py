"""Explanation generation for memories"""

from data_structures import Scene


class ExplanationGenerator:
    """Generates natural language explanations for memories"""

    @staticmethod
    def generate_explanation(
        scene: Scene,
        importance_score: float,
        motion_score: float,
        emotion_score: float,
        semantic_label: str
    ) -> str:
        """
        Generate a natural language explanation for why a scene/memory is important.

        Args:
            scene: Scene object with start and end time
            importance_score: float indicating importance
            motion_score: float indicating motion intensity (0-1)
            emotion_score: float indicating emotional intensity (0-1)
            semantic_label: string semantic label of the scene

        Returns:
            Explanation string
        """

        factors = []

        if motion_score > 0.6:
            factors.append("significant motion/activity")

        if emotion_score > 0.6:
            factors.append("high emotional intensity")

        if semantic_label == "important_explanation":
            factors.append("important concept introduction")
        elif semantic_label == "decision_moment":
            factors.append("key decision point")
        elif semantic_label == "transition":
            factors.append("scene change or transition")
        else:
            factors.append("notable content")

        # Join factors naturally
        factors_text = " and ".join(factors)

        duration = scene.end_time - scene.start_time
        explanation = f"This {duration:.1f}s moment is important because of: {factors_text}."

        return explanation
