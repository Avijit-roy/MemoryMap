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

        # ✅ Create heavy analyzers ONCE
        self.object_analyzer = ObjectContextAnalyzer()

    def run(
        self,
        sample_interval: float = 2.0,
        keep_ratio: float = 0.3,
        scene_threshold: float = 25.0,
    ):
        """
        Run complete memory extraction pipeline

        Args:
            sample_interval: seconds between frames
            keep_ratio: fraction of scenes to keep as memories
            scene_threshold: sensitivity for scene detection
        """

        print("\n" + "=" * 70)
        print("🧠 MEMORYMAP - VIDEO MEMORY EXTRACTION")
        print("=" * 70 + "\n")

        # 1️⃣ Video Ingestion
        print("1️⃣ Loading video...")
        video = VideoIngestion(self.video_path)
        metadata = video.get_metadata()

        print(f"  Duration: {metadata['duration_seconds']:.1f}s")
        print(f"  Resolution: {metadata['resolution']}")
        print(f"  FPS: {metadata['fps']}")

        # 2️⃣ Frame Sampling
        print("\n2️⃣ Sampling frames...")
        sampler = FrameSampling(video, sample_interval=sample_interval)
        frames = sampler.sample()

        # 3️⃣ Scene Segmentation
        print("\n3️⃣ Detecting motion events...")
        event_detector = MotionEventSegmentation(
            diff_threshold=15.0,
            min_event_frames=3,
            max_gap_frames=1
        )
        scenes = event_detector.detect_events(frames)


        # 4️⃣ Representative Frame Selection
        print("\n4️⃣ Selecting representative frames...")
        scenes = RepresentativeFrameSelection.process_all_scenes(scenes)

        # 5️⃣ Scene Analysis
        print("\n📊 Analyzing scenes...")
        scored_scenes = []

        for scene in scenes:
            # Object & Context Analysis (FIXED)
            try:
                context_data = self.object_analyzer.analyze_scene(scene)
            except Exception as e:
                print("⚠️ Object analysis failed:", e)
                context_data = {
                    "objects": [],
                    "object_count": 0,
                    "context_change": 0.0,
                }

            # Motion Analysis
            motion_score = MotionAnalysis.calculate_motion_score(scene)

            # Emotion / Visual Intensity Analysis
            emotion_score = EmotionAnalysis.estimate_emotion_score(scene)

            # Semantic Understanding
            semantic_label = SemanticAnalyzer.classify_scene(
                scene, motion_score, context_data["context_change"]
            )

            # Importance Scoring
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

        # 7️⃣ Explanation Generation & Timeline Output
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
