"""Importance scoring combining multiple signals (CCTV focused)"""

from data_structures import Scene

# Constants for importance scoring
MOTION_WEIGHT = 0.40
OBJECT_DELTA_WEIGHT = 0.30
DURATION_WEIGHT = 0.20
SUDDENNESS_WEIGHT = 0.10

DURATION_NORM = 10.0  # 10s ≈ full weight
MOTION_NORM = 25.0
SUDDENNESS_NORM = 20.0

# Semantic multipliers
SEMANTIC_MULTIPLIERS = {
    "idle_scene": 0.2,
    "minor_activity": 0.6,
    "significant_activity": 1.1,
    "critical_activity": 1.3,
    "new_object_appearance": 1.0,
    "background_activity": 0.8,
}

# Importance thresholds
IMPORTANCE_THRESHOLDS = {
    0.25: "Minor activity",
    0.55: "Significant disturbance",
    1.0: "Critical event"
}


class ImportanceScoringEngine:
    """Scores scene importance for static surveillance videos"""

    @staticmethod
    def score_scene(
        scene: Scene,
        context_data: dict,
        semantic_label: str
    ) -> dict:
        """
        Event-level importance scoring
        
        Returns:
            {
              "score": float (0–1),
              "level": str
            }
        """
        
        # Validate scene attributes exist
        motion_intensity = getattr(scene, 'motion_mean', 0.0)
        motion_peak = getattr(scene, 'motion_peak', 0.0)
        suddenness = getattr(scene, 'suddenness', 0.0)
        duration = getattr(scene, 'duration', 0.0)

        # Extract context signals
        object_delta = min(context_data.get("context_change", 0.0), 1.0)
        object_count = context_data.get("object_count", 0)

        # Normalize scores
        duration_score = min(duration / DURATION_NORM, 1.0)
        motion_score = min(motion_intensity / MOTION_NORM, 1.0)
        suddenness_score = min(suddenness / SUDDENNESS_NORM, 1.0)

        # Core importance formula
        importance = (
            MOTION_WEIGHT * motion_score +
            OBJECT_DELTA_WEIGHT * object_delta +
            DURATION_WEIGHT * duration_score +
            SUDDENNESS_WEIGHT * suddenness_score
        )

        # Semantic modulation
        multiplier = SEMANTIC_MULTIPLIERS.get(semantic_label, 1.0)
        importance *= multiplier

        # Penalize scenes with no objects
        if object_count == 0:
            importance *= 0.5

        # Clamp to valid range
        importance = max(0.0, min(importance, 1.0))

        # Determine importance level
        level = "Critical event"
        for threshold, label in sorted(IMPORTANCE_THRESHOLDS.items()):
            if importance < threshold:
                level = label
                break

        return {
            "score": importance,
            "level": level
        }


