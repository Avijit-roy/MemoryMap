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
            scene: Scene object
            importance_score: Importance score 0-1
            motion_score: Motion intensity 0-1
            emotion_score: Emotion/intensity score 0-1
            semantic_label: Semantic classification label
            
        Returns:
            str: Natural language explanation
        """
        factors = []

        # Motion-based explanation
        if motion_score > 0.6:
            factors.append("significant motion or activity detected")
        elif motion_score > 0.2:
            factors.append("noticeable movement")

        # Visual saliency/intensity
        if emotion_score > 0.6:
            factors.append("high visual intensity")

        # Semantic explanation (CCTV-aligned)
        semantic_explanations = {
            "new_object_appearance": "a new object appeared in the scene",
            "significant_activity": "sustained or strong activity occurred",
            "critical_activity": "critical activity was detected",
            "minor_activity": "minor activity was detected",
            "idle_scene": "the scene remained mostly idle",
            "background_activity": "background-level activity was present",
        }
        
        if semantic_label in semantic_explanations:
            factors.append(semantic_explanations[semantic_label])

        # Safety fallback
        if not factors:
            factors.append("notable visual changes")

        # Join factors naturally
        factors_text = " and ".join(factors)
        duration = scene.end_time - scene.start_time
        
        explanation = f"This {duration:.1f}s moment is important because {factors_text}."
        return explanation
