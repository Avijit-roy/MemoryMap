import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import json
from datetime import timedelta
import subprocess
import sys

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Frame:
    """Represents a single sampled frame"""
    image: np.ndarray
    timestamp: float
    frame_index: int

@dataclass
class Scene:
    """Represents a detected scene/event"""
    start_time: float
    end_time: float
    frames: List[Frame]
    representative_frame: np.ndarray
    scene_id: int

@dataclass
class Memory:
    """Represents a memory (selected scene with explanation)"""
    scene_id: int
    timestamp: float
    image_path: str
    importance_score: float
    explanation: str

# ============================================================================
# 1️⃣ VIDEO INGESTION MODULE
# ============================================================================

class VideoIngestion:
    """Handles video loading and metadata extraction"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def get_metadata(self) -> dict:
        """Returns video metadata"""
        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_seconds": self.duration,
            "width": self.width,
            "height": self.height,
            "resolution": f"{self.width}x{self.height}"
        }
    
    def read_frame(self) -> Tuple[bool, np.ndarray]:
        """Read next frame from video"""
        return self.cap.read()
    
    def seek_frame(self, frame_number: int):
        """Seek to specific frame"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    def close(self):
        """Close video stream"""
        self.cap.release()

# ============================================================================
# 2️⃣ FRAME SAMPLING MODULE
# ============================================================================

class FrameSampling:
    """Extracts frames at regular intervals"""
    
    def __init__(self, video_ingestion: VideoIngestion, sample_interval: float = 2.0):
        """
        Args:
            video_ingestion: VideoIngestion object
            sample_interval: seconds between sampled frames
        """
        self.video = video_ingestion
        self.sample_interval = sample_interval
        self.sampled_frames: List[Frame] = []
    
    def sample(self) -> List[Frame]:
        """Extract frames at regular intervals"""
        metadata = self.video.get_metadata()
        frame_interval = int(self.sample_interval * metadata["fps"])
        
        frame_index = 0
        sampled_count = 0
        
        while True:
            ret, frame = self.video.read_frame()
            if not ret:
                break
            
            if frame_index % frame_interval == 0:
                timestamp = frame_index / metadata["fps"]
                self.sampled_frames.append(Frame(
                    image=frame,
                    timestamp=timestamp,
                    frame_index=frame_index
                ))
                sampled_count += 1
            
            frame_index += 1
        
        print(f"✓ Sampled {sampled_count} frames at {self.sample_interval}s intervals")
        return self.sampled_frames

# ============================================================================
# 3️⃣ SCENE SEGMENTATION MODULE
# ============================================================================

class SceneSegmentation:
    """Detects scene boundaries based on frame differences"""
    
    def __init__(self, frames: List[Frame], threshold: float = 25.0):
        """
        Args:
            frames: List of sampled frames
            threshold: Color difference threshold for scene change
        """
        self.frames = frames
        self.threshold = threshold
        self.scenes: List[Scene] = []
    
    def detect_scenes(self) -> List[Scene]:
        """Detect scenes by comparing consecutive frames"""
        if len(self.frames) < 2:
            return []
        
        scene_starts = [0]
        
        for i in range(1, len(self.frames)):
            prev_frame = cv2.cvtColor(self.frames[i-1].image, cv2.COLOR_BGR2GRAY)
            curr_frame = cv2.cvtColor(self.frames[i].image, cv2.COLOR_BGR2GRAY)
            
            # Compute frame difference
            diff = cv2.absdiff(prev_frame, curr_frame)
            mean_diff = np.mean(diff)
            
            if mean_diff > self.threshold:
                scene_starts.append(i)
        
        # Create scenes from detected boundaries
        for scene_id, start_idx in enumerate(scene_starts):
            end_idx = scene_starts[scene_id + 1] if scene_id + 1 < len(scene_starts) else len(self.frames)
            
            scene_frames = self.frames[start_idx:end_idx]
            self.scenes.append(Scene(
                start_time=scene_frames[0].timestamp,
                end_time=scene_frames[-1].timestamp,
                frames=scene_frames,
                representative_frame=None,
                scene_id=scene_id
            ))
        
        print(f"✓ Detected {len(self.scenes)} scenes")
        return self.scenes

# ============================================================================
# 4️⃣ REPRESENTATIVE FRAME SELECTION
# ============================================================================

class RepresentativeFrameSelection:
    """Selects key frames from each scene"""
    
    @staticmethod
    def select_for_scene(scene: Scene) -> np.ndarray:
        """Select most representative frame from scene"""
        if not scene.frames:
            return None
        
        # Use middle frame as representative (simple approach)
        middle_idx = len(scene.frames) // 2
        return scene.frames[middle_idx].image
    
    @staticmethod
    def process_all_scenes(scenes: List[Scene]) -> List[Scene]:
        """Process all scenes to select representative frames"""
        for scene in scenes:
            scene.representative_frame = RepresentativeFrameSelection.select_for_scene(scene)
        print(f"✓ Selected representative frames for {len(scenes)} scenes")
        return scenes

# ============================================================================
# 5️⃣ OBJECT & CONTEXT UNDERSTANDING (Simplified)
# ============================================================================

class ObjectContextAnalyzer:
    """Analyzes objects and context changes"""
    
    @staticmethod
    def analyze_scene(scene: Scene) -> dict:
        """Analyze scene for objects and context changes"""
        # Simplified: detect if there's significant motion/change
        if len(scene.frames) < 2:
            return {"objects_detected": [], "context_change": 0.0}
        
        first = cv2.cvtColor(scene.frames[0].image, cv2.COLOR_BGR2GRAY)
        last = cv2.cvtColor(scene.frames[-1].image, cv2.COLOR_BGR2GRAY)
        
        change = np.mean(cv2.absdiff(first, last))
        change_normalized = min(change / 255.0, 1.0)
        
        return {
            "objects_detected": ["person", "text", "environment"],
            "context_change": change_normalized
        }

# ============================================================================
# 6️⃣ MOTION ANALYSIS
# ============================================================================

class MotionAnalysis:
    """Measures motion intensity in scenes"""
    
    @staticmethod
    def calculate_motion_score(scene: Scene) -> float:
        """Calculate motion score for scene (0-1)"""
        if len(scene.frames) < 2:
            return 0.0
        
        motion_scores = []
        
        for i in range(1, len(scene.frames)):
            prev = cv2.cvtColor(scene.frames[i-1].image, cv2.COLOR_BGR2GRAY)
            curr = cv2.cvtColor(scene.frames[i].image, cv2.COLOR_BGR2GRAY)
            
            # Optical flow approximation
            diff = np.mean(cv2.absdiff(prev, curr))
            motion_scores.append(diff)
        
        avg_motion = np.mean(motion_scores) if motion_scores else 0.0
        return min(avg_motion / 100.0, 1.0)  # Normalize

# ============================================================================
# 7️⃣ EMOTION & FACIAL ANALYSIS (Simplified)
# ============================================================================

class EmotionAnalysis:
    """Estimates emotional intensity in scenes"""
    
    @staticmethod
    def estimate_emotion_score(scene: Scene) -> float:
        """Estimate emotion intensity (0-1) based on frame variance"""
        if not scene.frames:
            return 0.0
        
        # Proxy: high entropy frames (more detail) suggest action/emotion
        frame = scene.representative_frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute Laplacian variance (focus measure)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        return min(variance / 500.0, 1.0)

# ============================================================================
# 8️⃣ SEMANTIC UNDERSTANDING (Simplified)
# ============================================================================

class SemanticAnalyzer:
    """Categorizes scenes semantically"""
    
    semantic_labels = [
        "important_explanation",
        "transition",
        "repetitive_content",
        "decision_moment",
        "introduction",
        "conclusion"
    ]
    
    @staticmethod
    def classify_scene(scene: Scene, motion_score: float, context_change: float) -> str:
        """Classify scene semantically"""
        if motion_score > 0.7:
            return "important_explanation"
        elif context_change > 0.6:
            return "transition"
        elif motion_score < 0.2:
            return "repetitive_content"
        else:
            return "decision_moment"

# ============================================================================
# 8️⃣ IMPORTANCE SCORING ENGINE
# ============================================================================

class ImportanceScoringEngine:
    """Combines all signals into importance score"""
    
    @staticmethod
    def score_scene(scene: Scene, 
                   motion_score: float,
                   emotion_score: float,
                   context_data: dict,
                   semantic_label: str) -> float:
        """
        Calculate importance score combining multiple signals
        Formula: Importance = motion + emotion + context_change + semantic_boost
        """
        # Base scores
        motion_weight = motion_score * 0.3
        emotion_weight = emotion_score * 0.25
        context_weight = context_data.get("context_change", 0.0) * 0.2
        
        # Semantic boost
        semantic_boost = 0.0
        if semantic_label in ["important_explanation", "decision_moment", "introduction"]:
            semantic_boost = 0.25
        elif semantic_label == "repetitive_content":
            semantic_boost = 0.0
        
        total_score = motion_weight + emotion_weight + context_weight + semantic_boost
        return min(total_score, 1.0)  # Normalize to 0-1

# ============================================================================
# 9️⃣ MEMORY SELECTION LOGIC
# ============================================================================

class MemorySelection:
    """Selects top-K scenes as memories"""
    
    @staticmethod
    def select_memories(scored_scenes: List[Tuple[Scene, float]], 
                       keep_ratio: float = 0.3) -> List[Tuple[Scene, float]]:
        """
        Select top memories, remove redundancy
        keep_ratio: fraction of scenes to keep as memories
        """
        # Sort by importance
        sorted_scenes = sorted(scored_scenes, key=lambda x: x[1], reverse=True)
        
        # Keep top K
        k = max(1, int(len(sorted_scenes) * keep_ratio))
        selected = sorted_scenes[:k]
        
        # Re-sort by timestamp for timeline
        selected = sorted(selected, key=lambda x: x[0].start_time)
        
        print(f"✓ Selected {len(selected)} memories from {len(sorted_scenes)} scenes")
        return selected

# ============================================================================
# 🔟 EXPLANATION GENERATOR
# ============================================================================

class ExplanationGenerator:
    """Generates natural language explanations for memories"""
    
    @staticmethod
    def generate_explanation(scene: Scene,
                            importance_score: float,
                            motion_score: float,
                            emotion_score: float,
                            semantic_label: str) -> str:
        """Generate explanation for why moment matters"""
        
        factors = []
        
        if motion_score > 0.6:
            factors.append("significant motion/activity")
        
        if emotion_score > 0.6:
            factors.append("high emotional intensity")
        
        if semantic_label == "important_explanation":
            factors.append("important concept introduction")
        elif semantic_label == "decision_moment":
            factors.append("key decision point")
        elif semantic_label == "transition":
            factors.append("scene change or transition")
        
        factors_text = " and ".join(factors) if factors else "notable content"
        
        duration = scene.end_time - scene.start_time
        explanation = f"This {duration:.1f}s moment is important because of: {factors_text}."
        
        return explanation

# ============================================================================
# 1️⃣1️⃣ MEMORY TIMELINE OUTPUT
# ============================================================================

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
    
    def save_timeline(self, memories: List[Memory]):
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

# ============================================================================
# MAIN PIPELINE
# ============================================================================

class MemoryMapPipeline:
    """Complete video memory extraction pipeline"""
    
    def __init__(self, video_path: str, output_dir: str = "memory_output"):
        self.video_path = video_path
        self.output_dir = output_dir
    
    def run(self, sample_interval: float = 2.0, 
            keep_ratio: float = 0.3,
            scene_threshold: float = 25.0):
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
        print("\n3️⃣ Detecting scenes...")
        detector = SceneSegmentation(frames, threshold=scene_threshold)
        scenes = detector.detect_scenes()
        
        # 4️⃣ Representative Frame Selection
        print("\n4️⃣ Selecting representative frames...")
        scenes = RepresentativeFrameSelection.process_all_scenes(scenes)
        
        # Analyze all scenes
        print("\n📊 Analyzing scenes...")
        scored_scenes = []
        
        for scene in scenes:
            # 5️⃣ Object & Context
            context_data = ObjectContextAnalyzer.analyze_scene(scene)
            
            # 6️⃣ Motion Analysis
            motion_score = MotionAnalysis.calculate_motion_score(scene)
            
            # 7️⃣ Emotion Analysis
            emotion_score = EmotionAnalysis.estimate_emotion_score(scene)
            
            # Semantic Understanding
            semantic_label = SemanticAnalyzer.classify_scene(scene, motion_score, context_data["context_change"])
            
            # 8️⃣ Importance Scoring
            importance_score = ImportanceScoringEngine.score_scene(
                scene, motion_score, emotion_score, context_data, semantic_label
            )
            
            scored_scenes.append((scene, importance_score, semantic_label, motion_score, emotion_score, context_data))
        
        # 9️⃣ Memory Selection
        print("\n🧠 Selecting memories...")
        selected = MemorySelection.select_memories(
            [(s, score) for s, score, _, _, _, _ in scored_scenes],
            keep_ratio=keep_ratio
        )
        
        # 🔟 Explanation Generation & 1️⃣1️⃣ Timeline Output
        print("\n📝 Generating explanations...")
        timeline = MemoryTimeline(self.output_dir)
        memories_with_explanations = []
        
        for selected_scene, importance_score in selected:
            # Find original analysis data
            analysis_data = next((d for s, _, _, _, _, d in scored_scenes if s == selected_scene), None)
            if not analysis_data:
                continue
            
            semantic_label = next((l for s, _, l, _, _, _ in scored_scenes if s == selected_scene), "unknown")
            motion_score = next((m for s, _, _, m, _, _ in scored_scenes if s == selected_scene), 0.0)
            emotion_score = next((e for s, _, _, _, e, _ in scored_scenes if s == selected_scene), 0.0)
            
            explanation = ExplanationGenerator.generate_explanation(
                selected_scene,
                importance_score,
                motion_score,
                emotion_score,
                semantic_label
            )
            
            memories_with_explanations.append((selected_scene, importance_score, explanation))
        
        timeline.save_timeline(memories_with_explanations)
        
        video.close()
        
        print("\n" + "=" * 70)
        print(f"✓ COMPLETE: Extracted {len(memories_with_explanations)} memories")
        print(f"📁 Output: {self.output_dir}/")
        print("=" * 70 + "\n")
        
        return memories_with_explanations

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python memorymap.py <video_path> [output_dir]")
        print("\nExample: python memorymap.py my_video.mp4 my_memories")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "memory_output"
    
    pipeline = MemoryMapPipeline(video_path, output_dir)
    pipeline.run(
        sample_interval=2.0,      # Sample every 2 seconds
        keep_ratio=0.3,            # Keep 30% of scenes as memories
        scene_threshold=25.0       # Scene change detection sensitivity
    )