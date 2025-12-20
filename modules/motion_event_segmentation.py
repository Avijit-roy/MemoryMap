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
        """
        Args:
            diff_threshold: motion sensitivity
            min_event_frames: minimum frames to form an event
            max_gap_frames: allowed silent frames inside an event
        """
        self.diff_threshold = diff_threshold
        self.min_event_frames = min_event_frames
        self.max_gap_frames = max_gap_frames

    def detect_events(self, frames: List[Frame]) -> List[Scene]:
        if len(frames) < 2:
            return []

        events: List[Scene] = []

        current_frames: List[Frame] = []
        silent_count = 0

        prev_gray = cv2.cvtColor(frames[0].image, cv2.COLOR_BGR2GRAY)

        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i].image, cv2.COLOR_BGR2GRAY)

            diff = cv2.absdiff(prev_gray, curr_gray)
            motion_energy = float(np.mean(diff))

            if motion_energy > self.diff_threshold:
                current_frames.append(frames[i])
                silent_count = 0
            else:
                if current_frames:
                    silent_count += 1
                    if silent_count <= self.max_gap_frames:
                        current_frames.append(frames[i])
                    else:
                        # finalize event
                        if len(current_frames) >= self.min_event_frames:
                            events.append(
                                Scene(
                                    frames=current_frames.copy(),
                                    start_time=current_frames[0].timestamp,
                                    end_time=current_frames[-1].timestamp
                                )
                            )
                        current_frames.clear()
                        silent_count = 0

            prev_gray = curr_gray

        # handle last event
        if len(current_frames) >= self.min_event_frames:
            events.append(
                Scene(
                    frames=current_frames,
                    start_time=current_frames[0].timestamp,
                    end_time=current_frames[-1].timestamp
                )
            )

        print(f"✓ Detected {len(events)} motion events")
        return events
