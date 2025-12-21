"""Data structures for MemoryMap pipeline"""

from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np


@dataclass
class Frame:
    """Represents a single sampled frame"""
    image: np.ndarray
    timestamp: float
    frame_index: int


@dataclass
class Scene:
    """Represents a detected scene/event"""

    scene_id: int
    start_time: float
    end_time: float
    frames: List[Frame]
    representative_frame: Optional[Union[Frame, np.ndarray]] = None


@dataclass
class Memory:
    """Represents a memory (selected scene with explanation)"""
    scene_id: int
    timestamp: float
    image_path: str
    importance_score: float
    explanation: str
