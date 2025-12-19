"""Main MemoryMap pipeline orchestration"""

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
        print("\n3️⃣ Detecting scenes...")
        detector = SceneSegmentationDL(self.video_path, threshold=scene_threshold)
        scenes = detector.detect_scenes(frames)

        # 4️⃣ Representative Frame Selection
        print("\n4️⃣ Selecting representative frames...")
        scenes = RepresentativeFrameSelection.process_all_scenes(scenes)

        # 5️⃣ Scene Analysis
        print("\n📊 Analyzing scenes...")
        scored_scenes = []
        scene_data = {}

        for scene in scenes:
            context_data = self.object_analyzer.analyze_scene(scene)
            motion_score = MotionAnalysis.calculate_motion_score(scene)
            emotion_score = EmotionAnalysis.estimate_emotion_score(scene)

            semantic_label = SemanticAnalyzer.classify_scene(
                scene,
                motion_score,
                context_data.get("context_change", 0.0)
            )

            importance_score = ImportanceScoringEngine.score_scene(
                scene,
                motion_score,
                emotion_score,
                context_data,
                semantic_label
            )

            scored_scenes.append((scene, importance_score))
            scene_data[id(scene)] = {
                "semantic": semantic_label,
                "motion": motion_score,
                "emotion": emotion_score
            }

        # 6️⃣ Memory Selection (Scene-based ✅)
        print("\n🧠 Selecting memories...")
        selected = MemorySelection.select_memories(
            scored_scenes,
            keep_ratio=keep_ratio
        )

        # 7️⃣ Explanation & Timeline
        print("\n📝 Generating explanations...")
        timeline = MemoryTimeline(self.output_dir)
        memories_with_explanations = []

        for scene, importance_score in selected:
            data = scene_data[id(scene)]

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
