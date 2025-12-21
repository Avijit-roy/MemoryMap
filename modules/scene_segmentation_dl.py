"""Scene segmentation using PySceneDetect (CCTV safe)"""

from typing import List
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from data_structures import Scene, Frame


class SceneSegmentationDL:
    """
    Scene segmentation using PySceneDetect

    - Safe fallback if detection fails
    - Works with sampled frames
    - Memory-efficient
    """

    def __init__(self, video_path: str, threshold: float = 27.0):
        self.video_path = video_path
        self.threshold = threshold

    def detect_scenes(self, frames: List[Frame]) -> List[Scene]:
        if not frames:
            return []

        # -------------------------
        # 1️⃣ Run PySceneDetect safely
        # -------------------------
        try:
            video = open_video(self.video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=self.threshold))
            scene_manager.detect_scenes(video)
            scene_list = scene_manager.get_scene_list()
        except Exception as e:
            print(f"⚠️ SceneDetect failed: {e}")
            scene_list = []

        # -------------------------
        # 2️⃣ Fallback: single scene
        # -------------------------
        if not scene_list:
            scene = Scene(
                start_time=frames[0].timestamp,
                end_time=frames[-1].timestamp,
                frames=frames,
                representative_frame=frames[len(frames) // 2]
            )
            return [scene]

        # -------------------------
        # 3️⃣ Map sampled frames to scenes (robust)
        # -------------------------
        scenes: List[Scene] = []

        for start, end in scene_list:
            start_time = start.get_seconds()
            end_time = end.get_seconds()

            scene_frames = [
                f for f in frames
                if f.timestamp >= start_time - 0.05
                and f.timestamp <= end_time + 0.05
            ]

            if not scene_frames:
                continue

            rep_frame = scene_frames[len(scene_frames) // 2]

            scenes.append(
                Scene(
                    start_time=start_time,
                    end_time=end_time,
                    frames=scene_frames,
                    representative_frame=rep_frame
                )
            )

        return scenes
