"""Main MemoryMap pipeline orchestration"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.video_ingestion import VideoIngestion
from modules.frame_sampling import FrameSampling
from modules.motion_event_segmentation import MotionEventSegmentation
from modules.representative_frames import RepresentativeFrameSelection
from modules.emotion_analysis import EmotionAnalysis
from modules.object_context import ObjectContextAnalyzer
from modules.semantic_analyzer import SemanticAnalyzer
from modules.importance_scoring import ImportanceScoringEngine
from modules.baseline_learner import CameraBaselinelearner  # ✅ NEW IMPORT
from modules.memory_selection import MemorySelection
from modules.explanation_generator import ExplanationGenerator
from modules.memory_timeline import MemoryTimeline


class MemoryMapPipeline:
    """Complete video memory extraction pipeline"""

    def __init__(self, video_path: str, output_dir: str = "memory_output"):
        self.video_path = video_path
        self.output_dir = output_dir
        self.object_analyzer = ObjectContextAnalyzer()

    def run(
        self,
        sample_interval: float = 1.0,
        keep_ratio: float = 0.2,
        adaptive_k: float = 2.5,
    ):
        print("\n" + "=" * 70)
        print("🧠 MEMORYMAP - VIDEO MEMORY EXTRACTION")
        print("=" * 70 + "\n")

        # 1. Load video
        print("1️⃣ Loading video...")
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"❌ Video file not found: {self.video_path}")

        video = VideoIngestion(self.video_path)
        metadata = video.get_metadata()

        print(f"  Duration: {metadata['duration_seconds']:.2f}s")
        print(f"  Resolution: {metadata['resolution']}")
        print(f"  FPS: {metadata['fps']:.2f}")

        # 2. Frame sampling
        print("\n2️⃣ Sampling frames...")
        frames = FrameSampling(video, sample_interval).sample()

        if not frames:
            print("⚠️ No frames extracted.")
            video.close()
            return []

        # 3. Motion event detection
        print("\n3️⃣ Detecting motion events...")
        detector = MotionEventSegmentation(
            min_event_frames=3,
            max_gap_frames=1,
            k=adaptive_k,
        )
        scenes = detector.detect_events(frames)

        if not scenes:
            print("⚠️ No motion scenes detected.")
            video.close()
            return []

        # 4. Representative frame selection
        print("\n4️⃣ Selecting representative frames...")
        scenes = RepresentativeFrameSelection.process_all_scenes(scenes)

        # 5. Analyze emotion for all scenes
        print("\n5️⃣ Analyzing emotion intensity...")
        for scene in scenes:
            scene.emotion_score = EmotionAnalysis.estimate_emotion_score(scene)

        # ✅ NEW: 6. Learn camera baseline from first third of scenes
        print("\n6️⃣ Learning camera baseline...")
        baseline_learner = CameraBaselinelearner(percentile=50)
        baseline = baseline_learner.learn_from_scenes(scenes[:len(scenes)//3])

        # 7. Scene analysis + importance scoring
        print("\n📊 Analyzing scenes...")
        scored_scenes = []

        for scene in scenes:
            try:
                context_data = self.object_analyzer.analyze_scene(scene)
            except Exception as e:
                print(f"⚠️ Object analysis failed: {e}")
                context_data = {
                    "objects": [],
                    "object_count": 0,
                    "context_change": 0.0,
                }

            semantic_label = SemanticAnalyzer.classify_scene(
                scene,
                scene.motion_mean,
                context_data.get("context_change", 0.0),
            )

            # ✅ NEW: Pass baseline to importance scoring
            importance = ImportanceScoringEngine.score_scene(
                scene,
                context_data,
                semantic_label,
                baseline=baseline,  # ✅ NEW PARAMETER
            )

            scored_scenes.append(
                (scene, importance, semantic_label)
            )

        # 8. Memory selection (score-only)
        print("\n🧠 Selecting memories...")
        selected = MemorySelection.select_memories(
            [(scene, imp["score"]) for scene, imp, _ in scored_scenes],
            keep_ratio=keep_ratio,
        )

        if not selected:
            print("⚠️ No memories selected.")
            video.close()
            return []

        # 9. Explanation + timeline
        print("\n📝 Generating explanations...")
        timeline = MemoryTimeline(self.output_dir)
        memories = []

        for selected_scene, score in selected:
            for scene, imp, label in scored_scenes:
                if scene == selected_scene:
                    # ✅ Pass all required parameters
                    explanation = ExplanationGenerator.generate_explanation(
                        scene,
                        imp["score"],
                        scene.motion_mean,
                        scene.emotion_score,
                        label,
                    )
                    memories.append((scene, imp["score"], explanation))
                    break

        timeline.save_timeline(memories)
        video.close()

        print("\n" + "=" * 70)
        print(f"✓ COMPLETE: Extracted {len(memories)} memories")
        print(f"📁 Output: {self.output_dir}/")
        print("=" * 70 + "\n")

        return memories