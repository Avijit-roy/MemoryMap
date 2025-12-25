import numpy as np
from typing import List, Dict
from data_structures import Scene

class CameraBaselinelearner:
    """Learn what's 'normal' for THIS specific camera"""
    
    def __init__(self, percentile: float = 50):
        """
        percentile: Use median (50th percentile) as normal baseline
        Use 25th percentile for "typical" vs 99th for "extreme"
        """
        self.percentile = percentile
        self.baseline = None
    
    def learn_from_scenes(self, scenes: List[Scene]) -> Dict[str, float]:
        """
        Analyze first N scenes to establish what's 'normal'
        Don't use labeled data—just learn the distribution
        """
        if len(scenes) < 10:
            print("⚠️ WARNING: <10 scenes. Baseline unreliable.")
        
        motion_values = [s.motion_mean for s in scenes]
        duration_values = [s.duration for s in scenes]
        suddenness_values = [s.suddenness for s in scenes]
        object_deltas = [getattr(s, 'object_delta', 0) for s in scenes]
        
        self.baseline = {
            "motion_mean": np.percentile(motion_values, self.percentile),
            "motion_std": np.std(motion_values),
            "motion_max": np.max(motion_values),
            "motion_min": np.min(motion_values),
            
            "duration_mean": np.percentile(duration_values, self.percentile),
            "duration_std": np.std(duration_values),
            "duration_max": np.max(duration_values),
            
            "suddenness_mean": np.percentile(suddenness_values, self.percentile),
            "suddenness_std": np.std(suddenness_values),
            
            "object_delta_mean": np.percentile(object_deltas, self.percentile),
            "scenes_used": len(scenes),
        }
        
        print(f"✓ Learned baseline from {len(scenes)} scenes")
        print(f"  Motion baseline: {self.baseline['motion_mean']:.2f} ± {self.baseline['motion_std']:.2f}")
        print(f"  Duration baseline: {self.baseline['duration_mean']:.2f}s ± {self.baseline['duration_std']:.2f}s")
        
        return self.baseline
    
    def get_baseline(self) -> Dict[str, float]:
        """Return learned baseline (or None if not learned yet)"""
        return self.baseline