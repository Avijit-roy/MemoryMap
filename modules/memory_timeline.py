import cv2
import json
from pathlib import Path
from typing import List, Tuple
from data_structures import Scene, Frame


class MemoryTimeline:
    """Generates and saves memory timeline"""

    def __init__(self, output_dir: str = "memory_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_timeline(self, memories: List[Tuple[Scene, float, str]]):
        """
        Save timeline images, JSON metadata and text report
        memories: List of (Scene, importance_score, explanation)
        """

        saved_memories = []

        # 1️⃣ Save images
        for idx, (scene, score, explanation) in enumerate(memories):
            img = None

            rf = scene.representative_frame
            if rf is not None:
                if isinstance(rf, Frame):
                    img = rf.image
                elif hasattr(rf, "shape"):
                    img = rf  # already ndarray

            if img is None:
                print(f"⚠️ Skipping memory {idx}: no representative frame")
                continue

            img_name = f"memory_{len(saved_memories):02d}.jpg"
            img_path = self.output_dir / img_name
            cv2.imwrite(str(img_path), img)

            saved_memories.append((scene, score, explanation, img_name))

        # 2️⃣ Save JSON timeline
        timeline_data = {
            "total_memories": len(saved_memories),
            "memories": [
                {
                    "index": idx,
                    "timestamp": self._format_time(scene.start_time),
                    "seconds": scene.start_time,
                    "importance": round(score, 3),
                    "explanation": explanation,
                    "image": img_name,
                }
                for idx, (scene, score, explanation, img_name)
                in enumerate(saved_memories)
            ],
        }

        with open(self.output_dir / "timeline.json", "w") as f:
            json.dump(timeline_data, f, indent=2)

        # 3️⃣ Save text report
        with open(self.output_dir / "memory_report.txt", "w") as f:
            f.write("=" * 70 + "\n")
            f.write("MEMORYMAP - VIDEO MEMORY EXTRACTION REPORT\n")
            f.write("=" * 70 + "\n\n")

            for idx, (scene, score, explanation, img_name) in enumerate(saved_memories):
                f.write(f"MEMORY #{idx + 1}\n")
                f.write(f"  Timestamp: {self._format_time(scene.start_time)}\n")
                f.write(f"  Importance Score: {score:.3f}\n")
                f.write(f"  Explanation: {explanation}\n")
                f.write(f"  Image: {img_name}\n\n")

        print(f"✓ Saved timeline to {self.output_dir}/")
        print("  - timeline.json")
        print("  - memory_report.txt")
        print(f"  - {len(saved_memories)} images")

    @staticmethod
    def _format_time(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"
