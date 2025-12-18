"""Frame sampling at regular intervals"""

from typing import List
from data_structures import Frame
from modules.video_ingestion import VideoIngestion


class FrameSampling:
    """Extracts frames at regular intervals"""
    
    def __init__(self, video_ingestion: VideoIngestion, sample_interval: float = 2.0):
        """
        Args:
            video_ingestion: VideoIngestion object
            sample_interval: seconds between sampled frames
        """
        self.video = video_ingestion
        self.sample_interval = sample_interval
        self.sampled_frames: List[Frame] = []
    
    def sample(self) -> List[Frame]:
        """Extract frames at regular intervals"""
        metadata = self.video.get_metadata()
        frame_interval = int(self.sample_interval * metadata["fps"])
        
        frame_index = 0
        sampled_count = 0
        
        while True:
            ret, frame = self.video.read_frame()
            if not ret:
                break
            
            if frame_index % frame_interval == 0:
                timestamp = frame_index / metadata["fps"]
                self.sampled_frames.append(Frame(
                    image=frame,
                    timestamp=timestamp,
                    frame_index=frame_index
                ))
                sampled_count += 1
            
            frame_index += 1
        
        print(f"✓ Sampled {sampled_count} frames at {self.sample_interval}s intervals")
        return self.sampled_frames