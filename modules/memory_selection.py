"""
Memory selection logic
"""

from typing import List, Tuple
from data_structures import Scene


class MemorySelection:
    """
    Selects top-K scenes as memories

    Works with (Scene, importance_score) tuples.
    Sorting by time is handled later in the pipeline.
    """

    @staticmethod
    def select_memories(
        scored_scenes: List[Tuple[Scene, float]],
        keep_ratio: float = 0.3
    ) -> List[Tuple[Scene, float]]:
        """
        Args:
            scored_scenes: List of (Scene, importance_score)
            keep_ratio: fraction of scenes to keep
        """

        if not scored_scenes:
            return []

        # 1️⃣ Sort by importance score (descending)
        sorted_scenes = sorted(
            scored_scenes,
            key=lambda x: x[1],
            reverse=True
        )

        # 2️⃣ Select top-K scenes
        k = max(1, int(len(sorted_scenes) * keep_ratio))
        selected = sorted_scenes[:k]

        print(f"✓ Selected {len(selected)} memories from {len(scored_scenes)} scenes")
        return selected
