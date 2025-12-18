"""Memory timeline generation and output"""

import cv2
import json
from pathlib import Path
from typing import List, Tuple
from data_structures import Scene, Memory


class MemoryTimeline:
    """Generates and saves memory timeline"""
    
    def __init__(self, output_dir: str = "memory_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.memories: List[Memory] = []
    
    def add_memory(self, scene: Scene, importance_score: float, 
                  explanation: str, image_filename: str):
        """Add memory to timeline"""
        memory = Memory(
            scene_id=scene.scene_id,
            timestamp=scene.start_time,
            image_path=image_filename,
            importance_score=importance_score,
            explanation=explanation
        )
        self.memories.append(memory)
    
    def save_timeline(self, memories: List[Tuple[Scene, float, str]]):
        """Save timeline as JSON and text"""
        # Save images
        for idx, (scene, score, explanation) in enumerate(memories):
            img_path = self.output_dir / f"memory_{idx:02d}.jpg"
            cv2.imwrite(str(img_path), scene.representative_frame)
        
        # Save JSON timeline
        timeline_data = {
            "total_memories": len(memories),
            "memories": [
                {
                    "index": idx,
                    "timestamp": f"{self._format_time(scene.start_time)}",
                    "seconds": scene.start_time,
                    "importance": f"{score:.2f}",
                    "explanation": explanation,
                    "image": f"memory_{idx:02d}.jpg"
                }
                for idx, (scene, score, explanation) in enumerate(memories)
            ]
        }
        
        json_path = self.output_dir / "timeline.json"
        with open(json_path, 'w') as f:
            json.dump(timeline_data, f, indent=2)
        
        # Save text report
        text_path = self.output_dir / "memory_report.txt"
        with open(text_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("MEMORYMAP - VIDEO MEMORY EXTRACTION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            for idx, (scene, score, explanation) in enumerate(memories):
                f.write(f"MEMORY #{idx + 1}\n")
                f.write(f"  Timestamp: {self._format_time(scene.start_time)}\n")
                f.write(f"  Importance Score: {score:.2%}\n")
                f.write(f"  Explanation: {explanation}\n")
                f.write(f"  Image: memory_{idx:02d}.jpg\n")
                f.write("\n")
        
        print(f"✓ Saved timeline to {self.output_dir}/")
        print(f"  - timeline.json")
        print(f"  - memory_report.txt")
        print(f"  - memory_*.jpg images")
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to MM:SS"""
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"