"""Importance scoring combining multiple signals"""

from data_structures import Scene


class ImportanceScoringEngine:
    """Combines all signals into importance score"""
    
    @staticmethod
    def score_scene(scene: Scene, 
                   motion_score: float,
                   emotion_score: float,
                   context_data: dict,
                   semantic_label: str) -> float:
        """
        Calculate importance score combining multiple signals
        Formula: Importance = motion + emotion + context_change + semantic_boost
        """
        # Base scores
        motion_weight = motion_score * 0.3
        emotion_weight = emotion_score * 0.25
        context_weight = context_data.get("context_change", 0.0) * 0.2
        
        # Semantic boost
        semantic_boost = 0.0
        if semantic_label in ["important_explanation", "decision_moment", "introduction"]:
            semantic_boost = 0.25
        elif semantic_label == "repetitive_content":
            semantic_boost = 0.0
        
        total_score = motion_weight + emotion_weight + context_weight + semantic_boost
        return min(total_score, 1.0)  # Normalize to 0-1