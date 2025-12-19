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