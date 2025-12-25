from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Union
import numpy as np

@dataclass
class Frame:
    """Frame data structure - add if missing"""
    image: Optional[np.ndarray] = None
    timestamp: float = 0.0
    frame_index: int = 0

@dataclass
class Scene:
    scene_id: int
    start_time: float
    end_time: float
    frames: List[Frame] = field(default_factory=list)
    representative_frame: Optional[Union[Frame, np.ndarray]] = None
    
    # Duration and motion metrics
    duration: float = 0.0
    motion_mean: float = 0.0      # ✅ ADDED - was missing
    motion_peak: float = 0.0       # ✅ ADDED - was missing
    motion_std: float = 0.0        # ✅ ADDED - was missing
    motion_intensity: float = 0.0
    suddenness: float = 0.0
    
    # Object/context metrics
    object_count_start: int = 0
    object_count_end: int = 0
    object_delta: int = 0
    
    # Importance metrics
    importance_score: float = 0.0
    importance_label: Optional[str] = None
    emotion_score: float = 0.0     # ✅ ADDED - for emotion analysis
