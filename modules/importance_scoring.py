"""Importance scoring combining multiple signals (CCTV focused)"""

import numpy as np
from typing import List
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


def _get_level(importance: float) -> str:
    """
    Determine importance level from score.
    
    Args:
        importance: Score between 0.0 and 1.0
        
    Returns:
        str: Level description
    """
    level = "Critical event"
    for threshold, label in sorted(IMPORTANCE_THRESHOLDS.items()):
        if importance < threshold:
            level = label
            break
    return level


class ImportanceScoringEngine:
    """Scores scene importance for static surveillance videos"""

    @staticmethod
    def score_scene(
        scene: Scene,
        context_data: dict,
        semantic_label: str,
        baseline: dict = None,  # ✅ NEW
    ) -> dict:
        """
        Score scene importance using deviation from camera baseline.
        
        This replaces hardcoded magic constants (divide by 25, divide by 10)
        with z-score deviation from THIS camera's statistical baseline.
        
        Args:
            scene: Scene object with motion/duration data
            context_data: Dict from ObjectContextAnalyzer (objects, context_change, etc)
            semantic_label: Scene classification (idle, minor, significant, critical)
            baseline: Camera baseline dict {motion_mean, motion_std, duration_mean, ...}
                     If None, uses reasonable defaults
                     
        Returns:
            dict: {
                "score": float (0-1),
                "level": str,
                "deviation_zscores": dict,
                "is_anomaly": bool,
                "confidence": str,
                "critical_objects": list,
                "has_weapon": bool,
            }
        """
        
        # ✅ NEW: Use per-camera baseline if provided
        if baseline is None:
            # Fallback: use reasonable defaults for unseen cameras
            baseline = {
                "motion_mean": 20.0,
                "motion_std": 10.0,
                "duration_mean": 5.0,
                "duration_std": 3.0,
                "suddenness_mean": 5.0,
                "suddenness_std": 3.0,
            }
        
        # Extract scene metrics (safe access with defaults)
        motion_intensity = getattr(scene, 'motion_mean', 0.0)
        duration = getattr(scene, 'duration', 0.0)
        suddenness = getattr(scene, 'suddenness', 0.0)
        object_count = getattr(scene, 'object_count', 0)
        
        # ✅ NEW: Calculate Z-score deviation from baseline
        # This tells us "how unusual is this compared to normal for THIS camera?"
        motion_zscore = (motion_intensity - baseline["motion_mean"]) / (baseline["motion_std"] + 1e-5)
        duration_zscore = (duration - baseline["duration_mean"]) / (baseline["duration_std"] + 1e-5)
        suddenness_zscore = (suddenness - baseline["suddenness_mean"]) / (baseline["suddenness_std"] + 1e-5)
        
        # ✅ NEW: Normalize z-scores to 0-1 range
        # 3σ = 1.0 (anything beyond 3 sigma gets clamped)
        motion_score = np.clip(motion_zscore / 3.0, 0, 1)
        duration_score = np.clip(duration_zscore / 3.0, 0, 1)
        suddenness_score = np.clip(suddenness_zscore / 3.0, 0, 1)
        
        # Object delta already normalized 0-1 from ObjectContextAnalyzer
        object_delta = min(context_data.get("context_change", 0.0), 1.0)
        
        # ✅ Weighted importance formula (same weights, but now properly normalized)
        importance = (
            MOTION_WEIGHT * motion_score +
            OBJECT_DELTA_WEIGHT * object_delta +
            DURATION_WEIGHT * duration_score +
            SUDDENNESS_WEIGHT * suddenness_score
        )
        
        # ✅ Semantic modulation (amplify/suppress based on event type)
        multiplier = SEMANTIC_MULTIPLIERS.get(semantic_label, 1.0)
        importance *= multiplier
        
        # ✅ Penalize scenes with no detected objects
        if object_count == 0:
            importance *= 0.5
        
        # ✅ Clamp to valid range
        importance = max(0.0, min(importance, 1.0))
        
        # ✅ NEW: Calculate confidence level
        confidence_level = _calculate_confidence(motion_zscore, duration_zscore)
        
        # ✅ NEW: Detect anomalies (2σ threshold)
        is_anomaly = motion_zscore > 2.0 or suddenness_zscore > 2.0
        
        # ✅ NEW: Extract additional context from context_data
        critical_objects = context_data.get("critical_objects", [])
        has_weapon = context_data.get("has_weapon", False)
        
        # ✅ NEW: Generate warnings for human review
        warnings = _generate_warnings(scene, baseline)
        
        return {
            "score": importance,
            "level": _get_level(importance),
            "confidence": confidence_level,
            "deviation_zscores": {
                "motion": round(motion_zscore, 3),
                "duration": round(duration_zscore, 3),
                "suddenness": round(suddenness_zscore, 3),
            },
            "reasons": {
                "high_motion": motion_zscore > 2.0,
                "unusual_duration": abs(duration_zscore) > 2.0,
                "sudden_change": suddenness_zscore > 2.0,
                "objects_changed": object_delta > 0.6,
                "critical_objects": critical_objects,
            },
            "is_anomaly": is_anomaly,
            "warnings": warnings,  # ✅ NEW: Human-readable warnings
            "critical_objects": critical_objects,  # Knife, gun, fire, etc
            "has_weapon": has_weapon,  # Boolean weapon flag
        }


def _calculate_confidence(motion_zscore: float, duration_zscore: float) -> str:
    """
    Calculate confidence level based on deviation magnitude.
    
    Args:
        motion_zscore: Z-score of motion intensity
        duration_zscore: Z-score of duration
        
    Returns:
        str: "LOW", "MEDIUM", or "HIGH"
    """
    max_zscore = max(abs(motion_zscore), abs(duration_zscore))
    
    if max_zscore < 1.0:
        return "LOW"  # Less than 1σ from baseline
    elif max_zscore < 2.0:
        return "MEDIUM"  # 1-2σ from baseline
    else:
        return "HIGH"  # Greater than 2σ from baseline (clear anomaly)


def _generate_warnings(scene: Scene, baseline: dict) -> List[str]:
    """
    Generate human-readable warnings based on anomalies detected.
    
    Args:
        scene: Scene object with motion/duration data
        baseline: Camera baseline dict
        
    Returns:
        list: List of warning strings for human review
    """
    warnings = []
    
    motion_intensity = getattr(scene, 'motion_mean', 0.0)
    duration = getattr(scene, 'duration', 0.0)
    object_count = getattr(scene, 'object_count', 0)
    
    # Check motion intensity
    if motion_intensity > baseline.get("motion_max", 100) * 1.5:
        warnings.append("⚠️ Motion intensity unusually high")
    
    # Check duration
    if duration > baseline.get("duration_max", 30) * 2:
        warnings.append("⚠️ Event duration longer than typical")
    
    # Check for false positives
    if object_count == 0:
        warnings.append("⚠️ Motion detected but no objects identified (possible false positive)")
    
    return warnings