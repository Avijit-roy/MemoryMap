"""
Entry point for MemoryMap application
"""

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
        sample_interval=1.0,   # temporal resolution
        keep_ratio=0.2,        # memory selectivity
        adaptive_k=2.5         # motion surprise sensitivity
    )


if __name__ == "__main__":
    main()
