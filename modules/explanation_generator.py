"""Explanation generation for memories"""

from data_structures import Scene


class ExplanationGenerator:
    """Generates natural language explanations for memories"""
    
    @staticmethod
    def generate_explanation(scene: Scene,
                            importance_score: float,
                            motion_score: float,
                            emotion_score: float,
                            semantic_label: str) -> str:
        """Generate explanation for why moment matters"""
        
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
        
        factors_text = " and ".join(factors) if factors else "notable content"
        
        duration = scene.end_time - scene.start_time
        explanation = f"This {duration:.1f}s moment is important because of: {factors_text}."
        
        return explanation