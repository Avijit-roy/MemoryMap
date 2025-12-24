import cv2
import json
from pathlib import Path
from typing import List, Tuple
from data_structures import Scene, Frame


def select_representative_frame(scene: Scene):
    """
    Select frame with highest motion inside the scene
    """
    if not scene.frames or len(scene.frames) < 2:
        return scene.frames[0].image if scene.frames else None

    best_frame = scene.frames[0].image
    best_motion = 0.0

    for i in range(len(scene.frames) - 1):
        f1 = scene.frames[i].image
        f2 = scene.frames[i + 1].image

        if f1 is None or f2 is None:
            continue

        gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray1, gray2)
        motion = diff.mean()

        if motion > best_motion:
            best_motion = motion
            best_frame = f2

    return best_frame


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
            # 🔥 ALWAYS recompute representative frame (fix randomness)
            img = select_representative_frame(scene)

            if img is None:
                print(f"⚠️ Skipping memory {idx}: no valid frame")
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
                    "seconds": round(scene.start_time, 2),
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
