"""Scene detection and segmentation"""

import cv2
import numpy as np
from typing import List
from data_structures import Frame, Scene


class SceneSegmentation:
    """Detects scene boundaries based on frame differences"""
    
    def __init__(self, frames: List[Frame], threshold: float = 25.0):
        """
        Args:
            frames: List of sampled frames
            threshold: Color difference threshold for scene change
        """
        self.frames = frames
        self.threshold = threshold
        self.scenes: List[Scene] = []
    
    def detect_scenes(self) -> List[Scene]:
        """Detect scenes by comparing consecutive frames"""
        if len(self.frames) < 2:
            return []
        
        scene_starts = [0]
        
        for i in range(1, len(self.frames)):
            prev_frame = cv2.cvtColor(self.frames[i-1].image, cv2.COLOR_BGR2GRAY)
            curr_frame = cv2.cvtColor(self.frames[i].image, cv2.COLOR_BGR2GRAY)
            
            # Compute frame difference
            diff = cv2.absdiff(prev_frame, curr_frame)
            mean_diff = np.mean(diff)
            
            if mean_diff > self.threshold:
                scene_starts.append(i)
        
        # Create scenes from detected boundaries
        for scene_id, start_idx in enumerate(scene_starts):
            end_idx = scene_starts[scene_id + 1] if scene_id + 1 < len(scene_starts) else len(self.frames)
            
            scene_frames = self.frames[start_idx:end_idx]
            self.scenes.append(Scene(
                start_time=scene_frames[0].timestamp,
                end_time=scene_frames[-1].timestamp,
                frames=scene_frames,
                representative_frame=None,
                scene_id=scene_id
            ))
        
        print(f"✓ Detected {len(self.scenes)} scenes")
        return self.scenes