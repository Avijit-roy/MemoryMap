"""
Deep-learning-based scene segmentation using PySceneDetect (PATCHED)
"""

from typing import List
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

from data_structures import Scene, Frame


class SceneSegmentationDL:
    """
    Scene segmentation using PySceneDetect

    PATCH:
    - No frame duplication
    - No storing raw video frames
    - Uses sampled frames only
    - Memory-safe for long videos
    """

    def __init__(self, video_path: str, threshold: float = 27.0):
        self.video_path = video_path
        self.threshold = threshold

    def detect_scenes(self, frames: List[Frame]) -> List[Scene]:
        """
        Detect scenes and map sampled frames into Scene objects
        """

        if not frames:
            return []

        # 1️⃣ Run PySceneDetect ONLY for timestamps
        video_manager = VideoManager([self.video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.threshold))

        try:
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)
            scene_list = scene_manager.get_scene_list()
        finally:
            video_manager.release()

        if not scene_list:
            # Fallback: single scene
            return [
                Scene(
                    frames=frames,
                    start_time=frames[0].timestamp,
                    end_time=frames[-1].timestamp
                )
            ]

        # 2️⃣ Map sampled frames → detected scenes
        scenes: List[Scene] = []

        for start, end in scene_list:
            start_time = start.get_seconds()
            end_time = end.get_seconds()

            scene_frames = [
                f for f in frames
                if start_time <= f.timestamp <= end_time
            ]

            if not scene_frames:
                continue

            scenes.append(
                Scene(
                    frames=scene_frames,
                    start_time=start_time,
                    end_time=end_time
                )
            )

        return scenes
