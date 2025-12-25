# 🧠 MemoryMap

**Intelligent Video Memory Extraction for Surveillance & Monitoring**

Extract the most important moments from video footage automatically. MemoryMap uses motion detection, object recognition, and AI-powered analysis to identify and summarize key scenes—perfect for security footage, time-lapse analysis, and video highlights.

![Status](https://img.shields.io/badge/status-active-green) ![Python](https://img.shields.io/badge/python-3.8+-blue) ![License](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Features

- 🎯 **Smart Motion Detection** - Adaptive motion-based event segmentation using K-sigma thresholds
- 🎬 **Scene Segmentation** - Automatic detection and isolation of distinct scenes
- 🔍 **Object Recognition** - YOLOv8-powered object detection (persons, vehicles, etc.)
- 💾 **Memory Selection** - Intelligent importance scoring to select top moments
- 📝 **Auto-Explanations** - Natural language descriptions of why each moment matters
- 📊 **Timeline Generation** - Visual and JSON output with memory metadata
- ⚡ **CCTV-Optimized** - Designed for static surveillance camera footage

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd memorymap

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
python main.py input_video.mp4 output_folder/
```

This will:
1. Extract frames from your video
2. Detect motion events
3. Analyze objects and context
4. Generate memory timeline
5. Save results to `output_folder/`

### Advanced Usage

```python
from pipeline import MemoryMapPipeline

pipeline = MemoryMapPipeline("video.mp4", "output/")
memories = pipeline.run(
    sample_interval=1.0,     # Frame sampling interval (seconds)
    keep_ratio=0.2,          # Keep top 20% of scenes
    adaptive_k=2.5           # Motion sensitivity (higher = stricter)
)
```

---

## 📋 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_interval` | 1.0 | Seconds between sampled frames (lower = more frames) |
| `keep_ratio` | 0.2 | Fraction of scenes to save as memories (0.0-1.0) |
| `adaptive_k` | 2.5 | Motion detection sensitivity (σ multiplier). Higher = fewer events detected |

---

## 📊 Pipeline Overview

```
Input Video
    ↓
1️⃣ Video Loading → Extract metadata (resolution, FPS, duration)
    ↓
2️⃣ Frame Sampling → Sample frames at regular intervals
    ↓
3️⃣ Motion Detection → Detect motion bursts as events
    ↓
4️⃣ Representative Frames → Select key frame for each scene
    ↓
5️⃣ Emotion Analysis → Calculate visual intensity scores
    ↓
6️⃣ Object Detection → Identify persons, vehicles, etc. (YOLO)
    ↓
7️⃣ Semantic Analysis → Classify event type (activity level)
    ↓
8️⃣ Importance Scoring → Calculate importance score for each scene
    ↓
9️⃣ Memory Selection → Select top-K memories by importance
    ↓
🔟 Timeline Generation → Generate JSON, images, and report
    ↓
Output Files
```

---

## 📂 Project Structure

```
memorymap/
├── main.py                          # Entry point
├── pipeline.py                      # Main orchestration
├── requirements.txt                 # Dependencies
├── modules/
│   ├── data_structures.py          # Scene, Frame dataclasses
│   ├── video_ingestion.py          # Video loading & metadata
│   ├── frame_sampling.py           # Frame extraction
│   ├── motion_event_segmentation.py # Motion-based event detection
│   ├── motion_analysis.py          # Motion intensity calculation
│   ├── representative_frames.py    # Key frame selection
│   ├── emotion_analysis.py         # Visual intensity scoring
│   ├── object_context.py           # YOLO object detection
│   ├── semantic_analyzer.py        # Event classification
│   ├── importance_scoring.py       # Memory importance calculation
│   ├── memory_selection.py         # Top-K memory selection
│   ├── explanation_generator.py    # Natural language generation
│   ├── memory_timeline.py          # Output generation
│   ├── utils.py                    # Helper functions
│   └── scene_segmentation_dl.py    # (Optional) PySceneDetect
└── memory_output/                   # Default output directory
    ├── timeline.json               # Memory metadata
    ├── memory_report.txt           # Text summary
    └── memory_*.jpg                # Representative images
```

---

## 📊 Output Files

### `timeline.json`
```json
{
  "total_memories": 5,
  "memories": [
    {
      "index": 0,
      "timestamp": "00:23",
      "seconds": 23.45,
      "importance": 0.856,
      "explanation": "This 4.2s moment is important because significant motion or activity detected and a new object appeared in the scene.",
      "image": "memory_00.jpg"
    }
  ]
}
```

### `memory_report.txt`
Text summary of all memories with timestamps, importance scores, and explanations.

### `memory_*.jpg`
Representative images from each important moment.

---

## 🔧 Configuration

### Tuning Parameters

**For more memories (keep more scenes):**
```python
keep_ratio=0.5  # Keep top 50% instead of 20%
```

**For stricter motion detection:**
```python
adaptive_k=3.5  # Only detect very obvious motion
```

**For more granular frame sampling:**
```python
sample_interval=0.5  # Sample every 0.5s instead of 1.0s
```

---

## 🧠 How It Works

### Motion-Based Event Detection
- Converts frames to grayscale and computes frame-to-frame differences
- Maintains adaptive baseline of motion history
- Detects motion "spikes" above mean + k×σ threshold
- Groups consecutive motion frames into events

### Importance Scoring Formula
```
importance = 0.40 × motion_score 
           + 0.30 × object_change
           + 0.20 × duration_score
           + 0.10 × suddenness_score
           × semantic_multiplier
```

**Semantic Multipliers:**
- Idle scene: 0.2× (less important)
- Minor activity: 0.6×
- Significant activity: 1.1× (more important)
- Critical activity: 1.3× (highest priority)

### Emotion/Visual Intensity
Combines:
- **Contrast** (60%): Standard deviation of grayscale values
- **Edge Density** (40%): Amount of edges detected (indicates structure)

---

## 🖥️ System Requirements

- **Python:** 3.8+
- **RAM:** 4GB minimum (8GB+ recommended)
- **GPU:** Optional (YOLO inference will be slower on CPU)
- **OS:** Linux, macOS, Windows

### Dependencies
```
opencv-python>=4.8.0
numpy>=1.21.0
PyAV>=10.0.0
ultralytics>=8.0.0  # YOLOv8
scenedetect>=0.6.1  # Optional
```

---

## 📦 Installation

### From Requirements File
```bash
pip install -r requirements.txt
```

### Manual Installation
```bash
pip install opencv-python numpy PyAV ultralytics scenedetect
```

### GPU Support (Optional)
```bash
# For CUDA GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🎯 Use Cases

- **Security Footage Analysis** - Highlight important events in surveillance videos
- **Time-Lapse Summarization** - Extract key moments from long recordings
- **Construction Monitoring** - Track project progress and identify issues
- **Wildlife Monitoring** - Detect and extract animal activity
- **Traffic Analysis** - Identify traffic incidents and congestion
- **Event Recording** - Automatically create highlight reels

---

## 🐛 Troubleshooting

### Issue: "No motion events detected"
**Solution:** Lower `adaptive_k` value (try 2.0 instead of 2.5)

### Issue: "YOLO model not loading"
**Solution:** Install ultralytics: `pip install ultralytics`

### Issue: "Out of memory"
**Solution:** Increase `sample_interval` (e.g., 2.0 instead of 1.0)

### Issue: "Corrupted MP4 file error"
**Solution:** Try converting video with FFmpeg first:
```bash
ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4
```

---

## 📈 Performance Tips

1. **Reduce frame sampling** for faster processing:
   ```python
   sample_interval=2.0  # Every 2 seconds instead of 1
   ```

2. **Use lower resolution video:**
   ```bash
   ffmpeg -i input.mp4 -vf scale=640:480 output.mp4
   ```

3. **Process only specific duration:**
   - Edit `video_ingestion.py` to limit duration

4. **Disable YOLO** if objects not needed:
   - Comment out object analysis in `pipeline.py`

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add multi-object tracking
- [ ] Implement face detection & recognition
- [ ] Add audio analysis
- [ ] Create web UI for visualization
- [ ] Add parallel processing
- [ ] Improve motion detection robustness

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👥 Authors

- MemoryMap Development Team

---

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check troubleshooting section above
- Review pipeline logs for detailed errors

---

## 🔄 Version History

### v1.0.0 (Current)
- ✅ Core motion detection pipeline
- ✅ Object recognition (YOLOv8)
- ✅ Importance scoring
- ✅ JSON & image output
- ✅ Emotion analysis integration

### Planned (v2.0)
- 🚧 Multi-object tracking
- 🚧 Web UI dashboard
- 🚧 Audio analysis
- 🚧 Parallel processing
- 🚧 Face detection

---

## 📚 Documentation

- **Pipeline Architecture** - See `pipeline.py`
- **Module Documentation** - See docstrings in each module
- **Data Structures** - See `data_structures.py`
- **Configuration** - See parameter tables above

---

## ⭐ Acknowledgments

Built with:
- [OpenCV](https://opencv.org/) - Image processing
- [YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [PyAV](https://pyav.org/) - Video decoding
- [NumPy](https://numpy.org/) - Numerical computing