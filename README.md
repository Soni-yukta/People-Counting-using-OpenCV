# People Counting & Spatial Flow Analytics using YOLOv8 & SORT

A production-grade computer vision and flow analytics system for pedestrian detection, multi-object tracking, and bi-directional flow counting using **YOLOv8**, a self-contained **Kalman Filter SORT** tracker, and an interactive **Gradio** web dashboard.

---

## Key Features

1. **State-of-the-Art Detection & Tracking**:
   - **YOLOv8 Nano** (`yolov8n.pt`) detection tuned for pedestrian recognition (COCO Class 0).
   - **Pure NumPy & SciPy SORT**: Custom Kalman filter state estimation with Hungarian algorithm data association, eliminating obsolete third-party dependencies (`filterpy`, `lap`, `skimage`).
   - Stateful centroid trajectory caching with hysteresis buffers preventing double counts and ghost triggers.

2. **Interactive Gradio Web Dashboard (`app.py`)**:
   - **Video Analytics Tab**: Upload MP4 or AVI videos, dynamically scale counting gates, display annotated playback, view real-time KPIs (IN, OUT, Net Occupancy, Unique Tracks, FPS), and download processed videos.
   - **Static Image Detection Tab**: Upload still images, detect individuals with customizable confidence thresholds, view bounding box badges, and review confidence metrics.
   - **Live Webcam Monitoring Tab**: Stream browser-based webcam frames with persistent tracker state across frames, directional gates, live HUD metrics, pause/resume toggling, and instant state reset.
   - **Visual Analytics**: Interactive BarPlot (IN vs OUT flow balance) and LinePlot (temporal occupancy curve).
   - **Audit Reports**: Download verified CSV event logs (`Event ID`, `Timestamp`, `Frame`, `Track ID`, `Direction`, `Net Occupancy`) and session summaries.

3. **High-Performance CLI Pipeline (`main.py`)**:
   - Flexible command-line arguments (`--source`, `--conf`, `--headless`, `--save-out`, `--max-frames`).
   - Zero per-frame disk I/O or terminal spam.
   - Graceful EOF handling and path-agnostic asset resolution.

---

## Directory Structure

```text
People-Counting-using-OpenCV-main/
├── .venv/                         # Python 3.11 Virtual Environment
├── app.py                         # Interactive Gradio Dashboard
├── main.py                        # CLI Video Processing Entry Point
├── sort.py                        # Self-contained SORT Multi-Object Tracker
├── utils.py                       # Geometry, LineGate, transparent overlay utilities
├── requirements.txt               # Pinned project dependencies
├── PROJECT_AUDIT.md               # Architecture audit and technical specifications
├── people.mp4                     # Sample escalator test video (1280x720)
├── mask-1.png                     # ROI escalator mask
├── graphics-1.png                 # Transparent HUD counter badge
├── yolov8n.pt                     # YOLOv8 Nano weights
├── reports/                       # Generated CSV audit logs
├── processed_videos/              # Exported annotated videos
├── test_phase2_verification.py    # Phase 2 Core CV test suite
├── test_phase3_dashboard.py       # Phase 3 Dashboard integration test suite
├── test_video_events_and_charts.py# Video line-crossing & charts test suite
└── test_server_live.py            # Local HTTP server probe test
```

---

## Quickstart Guide

### 1. Environment Setup (Python 3.11)

Activate the virtual environment:

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

Or install dependencies into your environment:
```bash
pip install -r requirements.txt
```

### 2. Launch the Interactive Web Dashboard

```powershell
python app.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:7860
```

### 3. Run via Command Line

Run inference with real-time OpenCV display:
```powershell
python main.py --source people.mp4
```

Run in headless mode and save annotated output video:
```powershell
python main.py --source people.mp4 --headless --save-out output_annotated.mp4
```

---

## Running Verification Tests

Run the full Phase 2 Core CV test suite:
```powershell
python test_phase2_verification.py
```

Run the Phase 3 Dashboard integration test suite:
```powershell
python test_phase3_dashboard.py
```

Run the live HTTP server probe test:
```powershell
python test_server_live.py
```

