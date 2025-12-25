"""Motion-based event segmentation for static surveillance videos"""

import cv2
import numpy as np
from typing import List
from collections import deque
from data_structures import Scene, Frame


# Configuration constants
MIN_EVENT_FRAMES = 3
MAX_GAP_FRAMES = 1
HISTORY_SIZE = 30
ADAPTIVE_K_DEFAULT = 2.5


class MotionEventSegmentation:
    """Detects motion bursts (events) from sampled frames"""

    def __init__(
        self,
        min_event_frames: int = MIN_EVENT_FRAMES,
        max_gap_frames: int = MAX_GAP_FRAMES,
        history_size: int = HISTORY_SIZE,
        k: float = ADAPTIVE_K_DEFAULT
    ):
        self.min_event_frames = min_event_frames
        self.max_gap_frames = max_gap_frames
        self.motion_history = deque(maxlen=history_size)
        self.k = k

    def detect_events(self, frames: List[Frame]) -> List[Scene]:
        """Detect motion events from frame list"""
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

            self.motion_history.append(motion_energy)

            # Learn baseline before detecting
            if len(self.motion_history) < 10:
                prev_gray = curr_gray
                continue

            mean = np.mean(self.motion_history)
            std = np.std(self.motion_history)
            adaptive_threshold = mean + self.k * std
            is_motion = motion_energy > adaptive_threshold

            if is_motion:
                if not current_frames:
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
                        self._finalize_event(events, current_frames, current_motion, scene_id)
                        if current_frames:
                            scene_id += 1
                        current_frames = []
                        current_motion = []
                        silent_count = 0

            prev_gray = curr_gray

        self._finalize_event(events, current_frames, current_motion, scene_id)
        print(f"✓ Detected {len(events)} motion events")
        return events

    def _finalize_event(
        self,
        events: List[Scene],
        frames: List[Frame],
        motion_values: List[float],
        scene_id: int
    ) -> None:
        """Finalize an event and add to events list"""
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
            duration=duration,
            motion_mean=motion_mean,
            motion_peak=motion_peak,
            motion_std=motion_std,
            suddenness=suddenness
        )

        events.append(scene)
