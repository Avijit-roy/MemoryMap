"""
Motion-based event segmentation for static surveillance videos
"""

import cv2
import numpy as np
from typing import List
from data_structures import Scene, Frame


class MotionEventSegmentation:
    """
    Detects motion bursts (events) from sampled frames

    An event = continuous period of motion
    """

    def __init__(
        self,
        diff_threshold: float = 15.0,
        min_event_frames: int = 3,
        max_gap_frames: int = 1
    ):
        self.diff_threshold = diff_threshold
        self.min_event_frames = min_event_frames
        self.max_gap_frames = max_gap_frames

    def detect_events(self, frames: List[Frame]) -> List[Scene]:
        if len(frames) < 2:
            return []

        events: List[Scene] = []

        current_frames: List[Frame] = []
        silent_count = 0
        scene_id = 0

        prev_gray = cv2.cvtColor(frames[0].image, cv2.COLOR_BGR2GRAY)

        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i].image, cv2.COLOR_BGR2GRAY)

            diff = cv2.absdiff(prev_gray, curr_gray)
            motion_energy = float(np.mean(diff))

            if motion_energy > self.diff_threshold:
                # ✅ include motion start frame
                if not current_frames:
                    current_frames.append(frames[i - 1])

                current_frames.append(frames[i])
                silent_count = 0
            else:
                if current_frames:
                    silent_count += 1
                    if silent_count <= self.max_gap_frames:
                        current_frames.append(frames[i])
                    else:
                        self._finalize_event(
                            events, current_frames, scene_id
                        )
                        if current_frames:
                            scene_id += 1
                        current_frames = []
                        silent_count = 0

            prev_gray = curr_gray

        # finalize last event
        self._finalize_event(events, current_frames, scene_id)

        print(f"✓ Detected {len(events)} motion events")
        return events

    def _finalize_event(
        self,
        events: List[Scene],
        frames: List[Frame],
        scene_id: int
    ):
        if len(frames) < self.min_event_frames:
            return

        events.append(
            Scene(
                scene_id=scene_id,
                start_time=frames[0].timestamp,
                end_time=frames[-1].timestamp,
                frames=frames.copy(),
                representative_frame=None
            )
        )
