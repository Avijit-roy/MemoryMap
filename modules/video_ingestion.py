"""
Video loading and metadata extraction using PyAV
(Hardened for CCTV / corrupted MP4s)
"""

from typing import Tuple, Optional
from pathlib import Path
import numpy as np
import av


class VideoIngestion:
    """Handles video loading and metadata extraction"""

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)

        # ---- File existence check ----
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        # ---- Open container safely ----
        try:
            self.container = av.open(str(self.video_path))
        except av.AVError as e:
            raise ValueError(
                f"Cannot open video (invalid or corrupted file): {self.video_path}\n"
                f"Reason: {e}"
            )

        # ---- Get video stream ----
        video_streams = [s for s in self.container.streams if s.type == "video"]
        if not video_streams:
            raise ValueError(f"No video stream found in file: {self.video_path}")

        self.stream = video_streams[0]

        # ---- FPS (fallback safe) ----
        if self.stream.average_rate:
            self.fps = float(self.stream.average_rate)
        else:
            self.fps = 25.0  # CCTV safe default

        # ---- Resolution ----
        self.width = self.stream.width or 0
        self.height = self.stream.height or 0

        # ---- Duration (CCTV-safe) ----
        if self.stream.duration and self.stream.time_base:
            self.duration = float(self.stream.duration * self.stream.time_base)
        else:
            self.duration = 0.0  # Unknown duration

        # ---- Frame count (often missing in CCTV) ----
        self.total_frames = self.stream.frames or 0

        # ---- Initialize decoder ----
        self._reset_decoder()

    # -------------------------------------------------

    def _reset_decoder(self):
        """Reset frame generator"""
        self.frame_generator = self.container.decode(video=self.stream.index)

    # -------------------------------------------------

    def get_metadata(self) -> dict:
        """Returns video metadata"""
        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_seconds": self.duration,
            "width": self.width,
            "height": self.height,
            "resolution": f"{self.width}x{self.height}",
        }

    # -------------------------------------------------

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame (OpenCV style)"""
        try:
            frame = next(self.frame_generator)
            img = frame.to_ndarray(format="bgr24")
            return True, img
        except StopIteration:
            return False, None
        except Exception:
            # Broken frame → skip
            return False, None

    # -------------------------------------------------

    def seek_frame(self, frame_number: int):
        """
        Seek approximately to a frame number
        (frame-accurate seeking is NOT guaranteed in CCTV videos)
        """
        if frame_number <= 0:
            self._reset_decoder()
            return

        timestamp = frame_number / self.fps

        try:
            self.container.seek(
                int(timestamp * av.time_base),
                any_frame=True,
                backward=True,
                stream=self.stream,
            )
            self._reset_decoder()
        except Exception:
            # If seek fails, reset decoding
            self._reset_decoder()

    # -------------------------------------------------

    def close(self):
        """Close video container"""
        try:
            self.container.close()
        except Exception:
            pass
