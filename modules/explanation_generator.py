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
        """

        factors = []

        # Motion-based explanation
        if motion_score > 0.6:
            factors.append("significant motion or activity detected")
        elif motion_score > 0.2:
            factors.append("noticeable movement")

        # Visual saliency (low priority in CCTV)
        if emotion_score > 0.6:
            factors.append("high visual intensity")

        # Semantic explanation (CCTV-aligned)
        if semantic_label == "new_object_appearance":
            factors.append("a new object appeared in the scene")
        elif semantic_label == "significant_activity":
            factors.append("sustained or strong activity occurred")
        elif semantic_label == "minor_activity":
            factors.append("minor activity was detected")
        elif semantic_label == "idle_scene":
            factors.append("the scene remained mostly idle")
        elif semantic_label == "background_activity":
            factors.append("background-level activity was present")

        # Safety fallback
        if not factors:
            factors.append("notable visual changes")

        # Join factors naturally
        factors_text = " and ".join(factors)

        duration = scene.end_time - scene.start_time
        explanation = (
            f"This {duration:.1f}s moment is important because {factors_text}."
        )

        return explanation
