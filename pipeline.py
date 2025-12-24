"""Main MemoryMap pipeline orchestration"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.video_ingestion import VideoIngestion
from modules.frame_sampling import FrameSampling
from modules.motion_event_segmentation import MotionEventSegmentation
from modules.representative_frames import RepresentativeFrameSelection
from modules.object_context import ObjectContextAnalyzer
from modules.motion_analysis import MotionAnalysis
from modules.emotion_analysis import EmotionAnalysis
from modules.semantic_analyzer import SemanticAnalyzer
from modules.importance_scoring import ImportanceScoringEngine
from modules.memory_selection import MemorySelection
from modules.explanation_generator import ExplanationGenerator
from modules.memory_timeline import MemoryTimeline


class MemoryMapPipeline:
    """Complete video memory extraction pipeline"""

    def __init__(self, video_path: str, output_dir: str = "memory_output"):
        self.video_path = video_path
        self.output_dir = output_dir

        # Heavy analyzers (load once)
        self.object_analyzer = ObjectContextAnalyzer()

    def run(
        self,
        sample_interval: float = 2.0,
        keep_ratio: float = 0.3,
        adaptive_k: float = 2.5,
    ):
        print("\n" + "=" * 70)
        print("🧠 MEMORYMAP - VIDEO MEMORY EXTRACTION")
        print("=" * 70 + "\n")

        # 1️⃣ Validate video path
        print("1️⃣ Loading video...")

        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"❌ Video file not found: {self.video_path}")

        try:
            video = VideoIngestion(self.video_path)
        except Exception as e:
            print("❌ Failed to open video.")
            print("Reason:", e)
            print("\n👉 Possible causes:")
            print(" - Corrupted MP4 (moov atom missing)")
            print(" - Incomplete download")
            print(" - Unsupported codec")
            print(" - Zero-byte file")
            return []

        metadata = video.get_metadata()
        print(f"  Duration: {metadata['duration_seconds']:.2f}s")
        print(f"  Resolution: {metadata['resolution']}")
        print(f"  FPS: {metadata['fps']:.2f}")

        # 2️⃣ Frame Sampling
        print("\n2️⃣ Sampling frames...")
        sampler = FrameSampling(video, sample_interval=sample_interval)
        frames = sampler.sample()

        if not frames:
            print("⚠️ No frames extracted. Aborting pipeline.")
            video.close()
            return []

        # 3️⃣ Motion Event Segmentation
        print("\n3️⃣ Detecting motion events...")
        event_detector = MotionEventSegmentation(
            min_event_frames=3,
            max_gap_frames=1,
            k=adaptive_k,
        )
        scenes = event_detector.detect_events(frames)

        if not scenes:
            print("⚠️ No motion scenes detected.")
            video.close()
            return []

        # 4️⃣ Representative Frame Selection
        print("\n4️⃣ Selecting representative frames...")
        scenes = RepresentativeFrameSelection.process_all_scenes(scenes)

        # 5️⃣ Scene Analysis
        print("\n📊 Analyzing scenes...")
        scored_scenes = []

        for scene in scenes:
            try:
                context_data = self.object_analyzer.analyze_scene(scene)
            except Exception as e:
                print("⚠️ Object analysis failed:", e)
                context_data = {
                    "objects": [],
                    "object_count": 0,
                    "context_change": 0.0,
                }

            motion_score = MotionAnalysis.calculate_motion_score(scene)
            emotion_score = EmotionAnalysis.estimate_emotion_score(scene)

            semantic_label = SemanticAnalyzer.classify_scene(
                scene, motion_score, context_data["context_change"]
            )

            importance_score = ImportanceScoringEngine.score_scene(
                scene,
                motion_score,
                emotion_score,
                context_data,
                semantic_label,
            )

            scored_scenes.append(
                (
                    scene,
                    importance_score,
                    semantic_label,
                    motion_score,
                    emotion_score,
                    context_data,
                )
            )

        # 6️⃣ Memory Selection
        print("\n🧠 Selecting memories...")
        selected = MemorySelection.select_memories(
            [(scene, score) for scene, score, *_ in scored_scenes],
            keep_ratio=keep_ratio,
        )

        if not selected:
            print("⚠️ No memories selected.")
            video.close()
            return []

        # 7️⃣ Explanation + Timeline
        print("\n📝 Generating explanations...")
        timeline = MemoryTimeline(self.output_dir)
        memories_with_explanations = []

        for selected_scene, importance_score in selected:
            for s, _, label, motion, emotion, _ in scored_scenes:
                if s == selected_scene:
                    semantic_label = label
                    motion_score = motion
                    emotion_score = emotion
                    break

            explanation = ExplanationGenerator.generate_explanation(
                selected_scene,
                importance_score,
                motion_score,
                emotion_score,
                semantic_label,
            )

            memories_with_explanations.append(
                (selected_scene, importance_score, explanation)
            )

        timeline.save_timeline(memories_with_explanations)
        video.close()

        print("\n" + "=" * 70)
        print(f"✓ COMPLETE: Extracted {len(memories_with_explanations)} memories")
        print(f"📁 Output: {self.output_dir}/")
        print("=" * 70 + "\n")

        return memories_with_explanations
# -------------------------------------------------