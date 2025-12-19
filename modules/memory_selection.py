"""Memory selection logic"""

from typing import List, Tuple, Union
from data_structures import Scene


class MemorySelection:
    """Selects top-K scenes as memories"""

    @staticmethod
    def select_memories(
        scored_scenes: List[Tuple[Union[Scene, int], float]],
        keep_ratio: float = 0.3
    ) -> List[Tuple[Union[Scene, int], float]]:
        """
        Select top memories based on importance score.

        scored_scenes: List of (Scene, score)
        keep_ratio: fraction of scenes to keep as memories
        """

        if not scored_scenes:
            return []

        # 1️⃣ Sort by importance score (descending)
        sorted_scenes = sorted(scored_scenes, key=lambda x: x[1], reverse=True)

        # 2️⃣ Keep top K
        k = max(1, int(len(sorted_scenes) * keep_ratio))
        selected = sorted_scenes[:k]

        # 3️⃣ Sort by timeline if Scene object exists
        def timeline_key(item):
            scene_or_idx = item[0]
            if isinstance(scene_or_idx, Scene):
                return scene_or_idx.start_time
            # fallback: preserve relative order
            return 0

        selected = sorted(selected, key=timeline_key)

        print(f"✓ Selected {len(selected)} memories from {len(sorted_scenes)} scenes")
        return selected
