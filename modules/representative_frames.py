"""Selection of representative frames from scenes"""

from typing import List, Optional
from data_structures import Scene, Frame


class RepresentativeFrameSelection:
    """Selects key frames from each scene"""

    @staticmethod
    def select_for_scene(scene: Scene) -> Optional[Frame]:
        """
        Select the most representative frame from a scene.

        Current strategy:
        - Picks the middle frame as a simple heuristic

        Args:
            scene: Scene object containing frames

        Returns:
            Frame object representing the scene, or None if no frames
        """
        if not scene.frames:
            return None

        middle_idx = len(scene.frames) // 2
        return scene.frames[middle_idx]

    @staticmethod
    def process_all_scenes(scenes: List[Scene]) -> List[Scene]:
        """
        Process all scenes to select their representative frames.

        Args:
            scenes: List of Scene objects

        Returns:
            The same list of Scene objects with `representative_frame` set
        """
        for scene in scenes:
            scene.representative_frame = RepresentativeFrameSelection.select_for_scene(scene)

        print(f"✓ Selected representative frames for {len(scenes)} scenes")
        return scenes
