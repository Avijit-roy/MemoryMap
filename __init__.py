"""
MemoryMap - Video Memory Extraction Pipeline

A comprehensive system for extracting and summarizing key moments from videos
using multiple analysis signals (motion, emotion, semantic understanding).
"""

__version__ = "3.0.0"
__author__ = "Avijit Roy"
__description__ = "Focused on surveillance video analysis and memory extraction."

# Import main classes for easy access
from pipeline import MemoryMapPipeline
from data_structures import Frame, Scene, Memory