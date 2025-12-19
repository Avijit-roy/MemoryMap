"""
MemoryMap - Video Memory Extraction Pipeline

A comprehensive system for extracting and summarizing key moments from videos
using multiple analysis signals (motion, emotion, semantic understanding).
"""

__version__ = "1.1.0"
__author__ = "Avijit Roy"
__description__ = "Video memory extraction and summarization tool"

# Import main classes for easy access
from pipeline import MemoryMapPipeline
from data_structures import Frame, Scene, Memory

# Import all modules for convenience
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

# Public API
__all__ = [
    # Main pipeline
    "MemoryMapPipeline",
    
    # Data structures
    "Frame",
    "Scene",
    "Memory",
    
    # Analysis modules
    "VideoIngestion",
    "FrameSampling",
    "SceneSegmentation",
    "RepresentativeFrameSelection",
    "ObjectContextAnalyzer",
    "MotionAnalysis",
    "EmotionAnalysis",
    "SemanticAnalyzer",
    "ImportanceScoringEngine",
    "MemorySelection",
    "ExplanationGenerator",
    "MemoryTimeline",
]
