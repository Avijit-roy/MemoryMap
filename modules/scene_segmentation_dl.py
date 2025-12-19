from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from typing import List
from data_structures import Scene, Frame
import cv2


class SceneSegmentationDL:
    """Scene segmentation using PySceneDetect"""

    def __init__(self, video_path: str, threshold: float = 27.0):
        self.video_path = video_path
        self.threshold = threshold

    def detect_scenes(self, sampled_frames: List[Frame]) -> List[Scene]:
        video_manager = VideoManager([self.video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.threshold))

        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        scene_list = scene_manager.get_scene_list()
        video_manager.release()

        scenes: List[Scene] = []

        for idx, (start, end) in enumerate(scene_list):
            start_sec = start.get_seconds()
            end_sec = end.get_seconds()

            frames_in_scene = [
                f for f in sampled_frames
                if start_sec <= f.timestamp <= end_sec
            ]

            if not frames_in_scene:
                continue

            scenes.append(Scene(
                start_time=start_sec,
                end_time=end_sec,
                frames=frames_in_scene,
                scene_id=idx
            ))

        print(f"✓ PySceneDetect detected {len(scenes)} scenes")
        return scenes
