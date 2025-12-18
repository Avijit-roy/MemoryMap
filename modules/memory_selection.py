"""Memory selection logic"""

from typing import List, Tuple
from data_structures import Scene


class MemorySelection:
    """Selects top-K scenes as memories"""
    
    @staticmethod
    def select_memories(scored_scenes: List[Tuple[Scene, float]], 
                       keep_ratio: float = 0.3) -> List[Tuple[Scene, float]]:
        """
        Select top memories, remove redundancy
        keep_ratio: fraction of scenes to keep as memories
        """
        # Sort by importance
        sorted_scenes = sorted(scored_scenes, key=lambda x: x[1], reverse=True)
        
        # Keep top K
        k = max(1, int(len(sorted_scenes) * keep_ratio))
        selected = sorted_scenes[:k]
        
        # Re-sort by timestamp for timeline
        selected = sorted(selected, key=lambda x: x[0].start_time)
        
        print(f"✓ Selected {len(selected)} memories from {len(sorted_scenes)} scenes")
        return selected