"""Video loading and metadata extraction"""

import cv2
from typing import Tuple
import numpy as np


class VideoIngestion:
    """Handles video loading and metadata extraction"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
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
        """Read next frame from video"""
        return self.cap.read()
    
    def seek_frame(self, frame_number: int):
        """Seek to specific frame"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    def close(self):
        """Close video stream"""
        self.cap.release()