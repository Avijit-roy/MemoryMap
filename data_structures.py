"""Data structures for MemoryMap pipeline"""

from dataclasses import dataclass
from typing import List
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
    start_time: float
    end_time: float
    frames: List[Frame]
    representative_frame: np.ndarray = None
    scene_id: int = 0


@dataclass
class Memory:
    """Represents a memory (selected scene with explanation)"""
    scene_id: int
    timestamp: float
    image_path: str
    importance_score: float
    explanation: str