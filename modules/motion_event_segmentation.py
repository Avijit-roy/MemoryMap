"""
Motion-based event segmentation for static surveillance videos
"""

import cv2
import numpy as np
from typing import List
from collections import deque
from data_structures import Scene, Frame


class MotionEventSegmentation:
    """
    Detects motion bursts (events) from sampled frames

    An event = motion spike relative to recent history
    """

    def __init__(
        self,
        min_event_frames: int = 3,
        max_gap_frames: int = 1,
        history_size: int = 30,
        k: float = 2.5
    ):
        self.min_event_frames = min_event_frames
        self.max_gap_frames = max_gap_frames

        # Adaptive motion parameters
        self.motion_history = deque(maxlen=history_size)
        self.k = k

    def detect_events(self, frames: List[Frame]) -> List[Scene]:
        if len(frames) < 2:
            return []

        events: List[Scene] = []

        current_frames: List[Frame] = []
        current_motion: List[float] = []

        silent_count = 0
        scene_id = 0

        prev_gray = cv2.cvtColor(frames[0].image, cv2.COLOR_BGR2GRAY)

        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i].image, cv2.COLOR_BGR2GRAY)

            diff = cv2.absdiff(prev_gray, curr_gray)
            motion_energy = float(np.mean(diff))

            # --- Adaptive baseline update ---
            self.motion_history.append(motion_energy)

            # Learn baseline first
            if len(self.motion_history) < 10:
                prev_gray = curr_gray
                continue

            mean = np.mean(self.motion_history)
            std = np.std(self.motion_history)
            adaptive_threshold = mean + self.k * std

            is_motion = motion_energy > adaptive_threshold

            if is_motion:
                if not current_frames:
                    # include frame before spike
                    current_frames.append(frames[i - 1])

                current_frames.append(frames[i])
                current_motion.append(motion_energy)
                silent_count = 0
            else:
                if current_frames:
                    silent_count += 1
                    if silent_count <= self.max_gap_frames:
                        current_frames.append(frames[i])
                        current_motion.append(motion_energy)
                    else:
                        self._finalize_event(
                            events,
                            current_frames,
                            current_motion,
                            scene_id
                        )
                        if current_frames:
                            scene_id += 1
                        current_frames = []
                        current_motion = []
                        silent_count = 0

            prev_gray = curr_gray

        # finalize last event
        self._finalize_event(events, current_frames, current_motion, scene_id)

        print(f"✓ Detected {len(events)} motion events")
        return events

    def _finalize_event(
        self,
        events: List[Scene],
        frames: List[Frame],
        motion_values: List[float],
        scene_id: int
    ):
        if len(frames) < self.min_event_frames:
            return

        duration = frames[-1].timestamp - frames[0].timestamp

        motion_mean = float(np.mean(motion_values))
        motion_peak = float(np.max(motion_values))
        motion_std = float(np.std(motion_values))
        suddenness = motion_peak - motion_mean

        scene = Scene(
            scene_id=scene_id,
            start_time=frames[0].timestamp,
            end_time=frames[-1].timestamp,
            frames=frames.copy(),
            representative_frame=None
        )

        # Attach motion intelligence (non-breaking extension)
        scene.motion_mean = motion_mean
        scene.motion_peak = motion_peak
        scene.motion_std = motion_std
        scene.suddenness = suddenness
        scene.duration = duration

        events.append(scene)
