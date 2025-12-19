"""Entry point for MemoryMap application"""

import sys
from pipeline import MemoryMapPipeline

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path> [output_dir]")
        print("\nExample: python main.py my_video.mp4 my_memories")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "memory_output"
    
    # Initialize and run pipeline
    pipeline = MemoryMapPipeline(video_path, output_dir)
    pipeline.run(
        sample_interval=4.0,  # Sample every 4 seconds
        keep_ratio=0.2,       # Keep top 20% of scenes as memories
        scene_threshold=25.0  # Scene change detection sensitivity
    )


if __name__ == "__main__":
    main()
