"""
Main MemoryMap pipeline orchestration (patched & stable)
"""

from modules.video_ingestion import VideoIngestion
from modules.frame_sampling import FrameSampling
from modules.scene_segmentation_dl import SceneSegmentationDL
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
        self.object_analyzer = ObjectContextAnalyzer()  # load once

    def run(
        self,
        sample_interval: float = 2.0,
        keep_ratio: float = 0.3,
        scene_threshold: float = 27.0
    ):
        print("\n" + "=" * 70)
        print("🧠 MEMORYMAP - VIDEO MEMORY EXTRACTION")
        print("=" * 70 + "\n")

        # 1️⃣ Video ingestion
        print("1️⃣ Loading video...")
        video = VideoIngestion(self.video_path)
        metadata = video.get_metadata()

        print(f"  Duration: {metadata['duration_seconds']:.1f}s")
        print(f"  Resolution: {metadata['resolution']}")
        print(f"  FPS: {metadata['fps']}")

        # 2️⃣ Frame sampling
        print("\n2️⃣ Sampling frames...")
        sampler = FrameSampling(video, sample_interval=sample_interval)
        frames = sampler.sample()

        # 3️⃣ Scene segmentation (DL)
        print("\n3️⃣ Detecting scenes...")
        detector = SceneSegmentationDL(
            self.video_path,
            threshold=scene_threshold
        )
        scenes = detector.detect_scenes(frames)

        if not scenes:
            print("⚠️ No scenes detected. Exiting.")
            video.close()
            return []

        # 4️⃣ Representative frames
        print("\n4️⃣ Selecting representative frames...")
        scenes = RepresentativeFrameSelection.process_all_scenes(scenes)

        # 5️⃣ Scene analysis
        print("\n📊 Analyzing scenes...")
        scored_scenes = []
        scene_lookup = {}

        for idx, scene in enumerate(scenes):
            # Object & context
            context_data = self.object_analyzer.analyze_scene(scene)

            # Motion
            motion_score = MotionAnalysis.calculate_motion_score(scene)

            # Emotion
            emotion_score = EmotionAnalysis.estimate_emotion_score(scene)

            # Semantic classification
            semantic_label = SemanticAnalyzer.classify_scene(
                scene,
                motion_score,
                context_data.get("context_change", 0.0)
            )

            # Importance score
            importance_score = ImportanceScoringEngine.score_scene(
                scene,
                motion_score,
                emotion_score,
                context_data,
                semantic_label
            )

            scored_scenes.append((idx, importance_score))
            scene_lookup[idx] = {
                "scene": scene,
                "semantic": semantic_label,
                "motion": motion_score,
                "emotion": emotion_score
            }

        # 6️⃣ Memory selection (index-based)
        print("\n🧠 Selecting memories...")
        selected = MemorySelection.select_memories(
            scored_scenes,
            keep_ratio=keep_ratio
        )

        # 7️⃣ Explanation & timeline
        print("\n📝 Generating explanations...")
        timeline = MemoryTimeline(self.output_dir)
        memories_with_explanations = []

        for idx, importance_score in selected:
            data = scene_lookup[idx]
            scene = data["scene"]

            explanation = ExplanationGenerator.generate_explanation(
                scene,
                importance_score,
                data["motion"],
                data["emotion"],
                data["semantic"]
            )

            memories_with_explanations.append(
                (scene, importance_score, explanation)
            )

        timeline.save_timeline(memories_with_explanations)
        video.close()

        print("\n" + "=" * 70)
        print(f"✓ COMPLETE: Extracted {len(memories_with_explanations)} memories")
        print(f"📁 Output: {self.output_dir}/")
        print("=" * 70 + "\n")

        return memories_with_explanations
