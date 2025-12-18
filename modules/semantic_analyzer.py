"""Semantic understanding and scene classification"""

from data_structures import Scene


class SemanticAnalyzer:
    """Categorizes scenes semantically"""
    
    semantic_labels = [
        "important_explanation",
        "transition",
        "repetitive_content",
        "decision_moment",
        "introduction",
        "conclusion"
    ]
    
    @staticmethod
    def classify_scene(scene: Scene, motion_score: float, context_change: float) -> str:
        """Classify scene semantically"""
        if motion_score > 0.7:
            return "important_explanation"
        elif context_change > 0.6:
            return "transition"
        elif motion_score < 0.2:
            return "repetitive_content"
        else:
            return "decision_moment"