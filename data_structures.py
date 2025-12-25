from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np

@dataclass
class Scene:
    scene_id: int
    start_time: float
    end_time: float
    frames: List[Frame]

    representative_frame: Optional[Union[Frame, np.ndarray]] = None

    duration: float = 0.0
    motion_intensity: float = 0.0
    suddenness: float = 0.0

    object_count_start: int = 0
    object_count_end: int = 0
    object_delta: int = 0

    importance_score: float = 0.0
    importance_label: Optional[str] = None
