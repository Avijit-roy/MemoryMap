"""
Memory selection logic (patched)
"""

from typing import List, Tuple


class MemorySelection:
    """
    Selects top-K scenes as memories

    PATCH:
    - Works with (scene_index, importance_score)
    - Does NOT assume Scene object inside this module
    - Sorting by time is handled in pipeline using scene_lookup
    """

    @staticmethod
    def select_memories(
        scored_scenes: List[Tuple[int, float]],
        keep_ratio: float = 0.3
    ) -> List[Tuple[int, float]]:
        """
        Args:
            scored_scenes: List of (scene_index, importance_score)
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

        # 2️⃣ Select top-K
        k = max(1, int(len(sorted_scenes) * keep_ratio))
        selected = sorted_scenes[:k]

        print(f"✓ Selected {len(selected)} memories from {len(scored_scenes)} scenes")
        return selected
