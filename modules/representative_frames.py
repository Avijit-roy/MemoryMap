"""Selection of representative frames from scenes"""

import numpy as np
from typing import List
from data_structures import Scene


class RepresentativeFrameSelection:
    """Selects key frames from each scene"""
    
    @staticmethod
    def select_for_scene(scene: Scene) -> np.ndarray:
        """Select most representative frame from scene"""
        if not scene.frames:
            return None
        
        # Use middle frame as representative (simple approach)
        middle_idx = len(scene.frames) // 2
        return scene.frames[middle_idx].image
    
    @staticmethod
    def process_all_scenes(scenes: List[Scene]) -> List[Scene]:
        """Process all scenes to select representative frames"""
        for scene in scenes:
            scene.representative_frame = RepresentativeFrameSelection.select_for_scene(scene)
        print(f"✓ Selected representative frames for {len(scenes)} scenes")
        return scenes