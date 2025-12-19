"""Video loading and metadata extraction using PyAV"""

from typing import Tuple
import numpy as np
import av


class VideoIngestion:
    """Handles video loading and metadata extraction"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.container = av.open(video_path)
        self.stream = self.container.streams.video[0]
        
        self.fps = float(self.stream.average_rate)
        self.width = self.stream.width
        self.height = self.stream.height
        self.total_frames = self.stream.frames or 0
        self.duration = float(self.stream.duration * self.stream.time_base) if self.stream.duration else 0
        
        # Initialize frame generator
        self.frame_generator = self.container.decode(video=0)
    
    def get_metadata(self) -> dict:
        """Returns video metadata"""
        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_seconds": self.duration,
            "width": self.width,
            "height": self.height,
            "resolution": f"{self.width}x{self.height}"
        }
    
    def read_frame(self) -> Tuple[bool, np.ndarray]:
        """
        Read next frame in OpenCV style (ret, frame)
        """
        try:
            frame = next(self.frame_generator)
            img = frame.to_ndarray(format='bgr24')  # OpenCV compatible
            return True, img
        except StopIteration:
            return False, None
    
    def seek_frame(self, frame_number: int):
        """
        Seek to a specific frame number
        """
        timestamp = frame_number / self.fps
        self.container.seek(int(timestamp / self.stream.time_base))
        # Reset generator after seeking
        self.frame_generator = self.container.decode(video=0)
    
    def close(self):
        """Close video container"""
        self.container.close()
