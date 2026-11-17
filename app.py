"""
People Counting using YOLOv8 & SORT - Interactive Gradio Web Dashboard.
Includes Video Processing, Image Detection, Live Webcam Streaming, Real-time Analytics, and CSV Reporting.
"""

import os
import sys

# Prevent analytics and external font hang in isolated/sandboxed environments
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import gradio as gr

from sort import Sort
import utils


# ============================================================================
# 1. Model Singleton & Global Configuration
# ============================================================================

_YOLO_MODEL = None

def get_yolo_model():
    """Lazy singleton to ensure YOLO model is initialized only once."""
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        model_path = utils.resolve_asset_path("yolov8n.pt")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading YOLOv8 model from: {model_path} on {device}")
        _YOLO_MODEL = YOLO(model_path)
    return _YOLO_MODEL


REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

PROCESSED_VIDEO_DIR = Path("processed_videos")
PROCESSED_VIDEO_DIR.mkdir(exist_ok=True)


# ============================================================================
# 2. Video Processing Pipeline
# ============================================================================

def process_video(video_path, conf_threshold, max_frames, apply_mask, progress=gr.Progress()):
    """
    Processes an uploaded video through the verified YOLOv8 + SORT + LineGate pipeline.
    Returns:
      - Annotated video path (for playback and download)
      - Metrics: Total IN, Total OUT, Net Occupancy, Total Unique Tracks, Avg FPS
      - Bar chart: IN vs OUT comparison
      - Line chart: Occupancy trend over time
      - Events DataFrame
      - CSV report file path
      - Text analytics summary
    """
    if not video_path or not os.path.exists(video_path):
        return (None, 0, 0, 0, 0, 0.0, None, None, None, None, "Error: No valid video file provided.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return (None, 0, 0, 0, 0, 0.0, None, None, None, None, f"Error: Cannot open video '{video_path}'.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        total_frames = 1000  # Fallback estimate if container metadata missing

    effective_max_frames = int(max_frames) if max_frames > 0 else total_frames

    # Dynamic line scaling based on resolution
    scale_x = width / 1280.0
    scale_y = height / 720.0
    limits_up = [int(103 * scale_x), int(161 * scale_y), int(296 * scale_x), int(161 * scale_y)]
    limits_down = [int(527 * scale_x), int(489 * scale_y), int(735 * scale_x), int(489 * scale_y)]

    # Load model and assets
    model = get_yolo_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mask = None
    if apply_mask:
        mask_path = utils.resolve_asset_path("mask-1.png")
        if os.path.exists(mask_path):
            raw_mask = cv2.imread(mask_path)
            if raw_mask is not None:
                mask = cv2.resize(raw_mask, (width, height))

    graphics = None
    graphics_path = utils.resolve_asset_path("graphics-1.png")
    if os.path.exists(graphics_path):
        graphics = utils.load_overlay_image(graphics_path)

    # Initialize Tracker and Stateful Gates
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    tracker.reset()

    gate_up = utils.LineGate("IN/UP", limits_up, direction='up', hysteresis=int(15 * scale_y))
    gate_down = utils.LineGate("OUT/DOWN", limits_down, direction='down', hysteresis=int(15 * scale_y))

    # Output video writer
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_video_name = f"processed_{timestamp_str}.mp4"
    out_video_path = str(PROCESSED_VIDEO_DIR / out_video_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    # Data collection containers
    events_log = []
    trend_samples = []
    unique_ids = set()

    frame_idx = 0
    start_time = time.time()

    progress(0, desc="Starting video processing...")

    while True:
        success, img = cap.read()
        if not success or img is None:
            break

        frame_idx += 1
        if frame_idx > effective_max_frames:
            break

        # Progress reporting in Gradio
        if frame_idx % 10 == 0:
            fraction = min(1.0, frame_idx / effective_max_frames)
            progress(fraction, desc=f"Processing frame {frame_idx}/{effective_max_frames} ({(fraction*100):.0f}%)...")

        # Masking
        if mask is not None:
            img_region = cv2.bitwise_and(img, mask)
        else:
            img_region = img

        # Graphics banner
        if graphics is not None:
            img = utils.overlay_transparent(img, graphics, int(730 * scale_x), int(260 * scale_y))

        # YOLO detection
        results = model.predict(img_region, classes=[0], conf=conf_threshold, verbose=False, device=device)

        detections = np.empty((0, 5))
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                detections = np.vstack((detections, [int(x1), int(y1), int(x2), int(y2), conf]))

        # SORT update
        tracked_objects = tracker.update(detections)

        # Draw counting lines with active highlight
        color_up = (0, 255, 0) if gate_up.highlight_frames > 0 else (0, 0, 255)
        color_down = (0, 255, 0) if gate_down.highlight_frames > 0 else (0, 0, 255)
        if gate_up.highlight_frames > 0:
            gate_up.highlight_frames -= 1
        if gate_down.highlight_frames > 0:
            gate_down.highlight_frames -= 1

        cv2.line(img, (limits_up[0], limits_up[1]), (limits_up[2], limits_up[3]), color_up, 4)
        cv2.line(img, (limits_down[0], limits_down[1]), (limits_down[2], limits_down[3]), color_down, 4)

        # Tracked objects processing
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
            unique_ids.add(track_id)
            w, h = x2 - x1, y2 - y1

            utils.draw_tracker_box(img, x1, y1, x2, y2, track_id, color=(255, 0, 255))
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            trk = tracker.get_track_by_id(track_id)
            p_prev, p_curr = trk.get_last_two_centroids() if trk else ((cx, cy), (cx, cy))

            # Check crossings
            elapsed_sec = round(frame_idx / fps, 2)
            if gate_up.check_crossing(track_id, p_prev, p_curr):
                cv2.line(img, (limits_up[0], limits_up[1]), (limits_up[2], limits_up[3]), (0, 255, 0), 4)
                occ = max(0, gate_up.total_count - gate_down.total_count)
                events_log.append({
                    "Event ID": len(events_log) + 1,
                    "Timestamp (s)": elapsed_sec,
                    "Frame": frame_idx,
                    "Track ID": track_id,
                    "Direction": "IN (UP)",
                    "Net Occupancy": occ
                })

            if gate_down.check_crossing(track_id, p_prev, p_curr):
                cv2.line(img, (limits_down[0], limits_down[1]), (limits_down[2], limits_down[3]), (0, 255, 0), 4)
                occ = max(0, gate_up.total_count - gate_down.total_count)
                events_log.append({
                    "Event ID": len(events_log) + 1,
                    "Timestamp (s)": elapsed_sec,
                    "Frame": frame_idx,
                    "Track ID": track_id,
                    "Direction": "OUT (DOWN)",
                    "Net Occupancy": occ
                })

        # Render counter numbers onto graphics
        count_up_str = str(gate_up.total_count)
        count_down_str = str(gate_down.total_count)
        tx_up = int(929 * scale_x)
        ty_up = int(345 * scale_y)
        tx_down = int(1191 * scale_x)
        ty_down = int(345 * scale_y)
        cv2.putText(img, count_up_str, (tx_up, ty_up), cv2.FONT_HERSHEY_PLAIN, max(2.0, 5.0 * scale_y), (139, 195, 75), max(2, int(7 * scale_y)))
        cv2.putText(img, count_down_str, (tx_down, ty_down), cv2.FONT_HERSHEY_PLAIN, max(2.0, 5.0 * scale_y), (50, 50, 230), max(2, int(7 * scale_y)))

        writer.write(img)

        # Trend sampling every 10 frames
        if frame_idx % 10 == 0 or frame_idx == effective_max_frames:
            cur_occ = max(0, gate_up.total_count - gate_down.total_count)
            t_sec = round(frame_idx / fps, 1)
            trend_samples.append({
                "Time (s)": t_sec,
                "IN": gate_up.total_count,
                "OUT": gate_down.total_count,
                "Occupancy": cur_occ
            })

    cap.release()
    writer.release()

    total_processing_time = time.time() - start_time
    avg_fps = round(frame_idx / total_processing_time, 1) if total_processing_time > 0 else 0.0

    total_in = gate_up.total_count
    total_out = gate_down.total_count
    net_occupancy = max(0, total_in - total_out)
    unique_count = len(unique_ids)

    # DataFrames for UI and charts
    events_df = pd.DataFrame(events_log)
    if events_df.empty:
        events_df = pd.DataFrame(columns=["Event ID", "Timestamp (s)", "Frame", "Track ID", "Direction", "Net Occupancy"])

    trend_df = pd.DataFrame(trend_samples)
    if trend_df.empty:
        trend_df = pd.DataFrame([{"Time (s)": 0.0, "IN": 0, "OUT": 0, "Occupancy": 0}])

    in_out_df = pd.DataFrame([
        {"Category": "Total IN", "Count": total_in},
        {"Category": "Total OUT", "Count": total_out},
        {"Category": "Net Occupancy", "Count": net_occupancy}
    ])

    # Save CSV Report
    csv_report_path = str(REPORTS_DIR / f"counting_events_{timestamp_str}.csv")
    events_df.to_csv(csv_report_path, index=False)

    summary_text = f"""### Processing Summary
- **Source File:** `{os.path.basename(video_path)}` ({width}x{height} @ {fps:.1f} FPS)
- **Processed Frames:** {frame_idx} / {total_frames} (Execution time: {total_processing_time:.2f}s)
- **Average Performance:** **{avg_fps} FPS** ({device.upper()} acceleration)
- **Total Inflow (IN):** **{total_in}**
- **Total Outflow (OUT):** **{total_out}**
- **Current Net Occupancy:** **{net_occupancy}**
- **Total Unique People Identified:** **{unique_count}**
- **CSV Event Log Saved:** `{os.path.basename(csv_report_path)}`
"""

    return (
        out_video_path,
        total_in,
        total_out,
        net_occupancy,
        unique_count,
        avg_fps,
        in_out_df,
        trend_df,
        events_df,
        csv_report_path,
        summary_text
    )


# ============================================================================
# 3. Image Detection Pipeline
# ============================================================================

def process_image(input_img, conf_threshold):
    """
    Detects people in an uploaded static image using YOLOv8.
    Returns:
      - Annotated image
      - People count
      - Statistics (Average Conf, Min Conf, Max Conf)
      - Detailed detection table
    """
    if input_img is None:
        return (None, 0, "No image uploaded", pd.DataFrame())

    # Input comes as RGB NumPy array from Gradio
    img_bgr = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    model = get_yolo_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = model.predict(img_bgr, classes=[0], conf=conf_threshold, verbose=False, device=device)

    detections = []
    confs = []

    for r in results:
        for idx, box in enumerate(r.boxes, start=1):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confs.append(conf)

            # Draw box and confidence badge
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 255), 2)
            label = f"Person #{idx}: {conf*100:.1f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), (255, 0, 255), -1)
            cv2.putText(img_bgr, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            detections.append({
                "Person #": idx,
                "Confidence": f"{conf*100:.1f}%",
                "Bounding Box": f"[{x1}, {y1}, {x2}, {y2}]",
                "Width (px)": x2 - x1,
                "Height (px)": y2 - y1
            })

    annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    total_people = len(detections)

    if confs:
        stats = f"Detected **{total_people} people**. Confidence: Avg={np.mean(confs)*100:.1f}%, Min={np.min(confs)*100:.1f}%, Max={np.max(confs)*100:.1f}%"
    else:
        stats = "No people detected above the confidence threshold."

    df = pd.DataFrame(detections)
    if df.empty:
        df = pd.DataFrame(columns=["Person #", "Confidence", "Bounding Box", "Width (px)", "Height (px)"])

    return annotated_rgb, total_people, stats, df


# ============================================================================
# 4. Live Webcam State & Streaming Pipeline
# ============================================================================

class WebcamStreamState:
    """Preserves tracker, gates, and count state across webcam frames."""
    def __init__(self):
        self.tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)
        self.tracker.reset()
        self.gate_in = None
        self.gate_out = None
        self.is_active = True
        self.frame_idx = 0
        self.start_time = time.time()
        self.fps_tracker = []
        self.unique_ids = set()

    def reset(self):
        self.tracker.reset()
        if self.gate_in:
            self.gate_in.reset()
        if self.gate_out:
            self.gate_out.reset()
        self.frame_idx = 0
        self.start_time = time.time()
        self.fps_tracker.clear()
        self.unique_ids.clear()


# Global or per-session webcam stream instance
_WEBCAM_STATE = WebcamStreamState()

def process_webcam_frame(input_frame, conf_threshold):
    """
    Processes a single live webcam frame preserving tracker and count state.
    """
    global _WEBCAM_STATE
    if input_frame is None:
        return None, 0, 0, 0, 0, 0.0

    state = _WEBCAM_STATE
    if not state.is_active:
        return input_frame, (state.gate_in.total_count if state.gate_in else 0), \
               (state.gate_out.total_count if state.gate_out else 0), 0, 0, 0.0

    t_start = time.time()
    state.frame_idx += 1

    # Convert RGB from Gradio to BGR
    img = cv2.cvtColor(input_frame, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    # Initialize gates on first frame or if size changes
    if state.gate_in is None or state.gate_in.line_coords[2] != w:
        line_in = [int(w * 0.1), int(h * 0.4), int(w * 0.9), int(h * 0.4)]
        line_out = [int(w * 0.1), int(h * 0.65), int(w * 0.9), int(h * 0.65)]
        state.gate_in = utils.LineGate("IN", line_in, direction='down', hysteresis=15)
        state.gate_out = utils.LineGate("OUT", line_out, direction='up', hysteresis=15)

    model = get_yolo_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = model.predict(img, classes=[0], conf=conf_threshold, verbose=False, device=device)

    detections = np.empty((0, 5))
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            detections = np.vstack((detections, [int(x1), int(y1), int(x2), int(y2), conf]))

    tracked_objects = state.tracker.update(detections)

    # Draw counting lines with active flash
    col_in = (0, 255, 0) if state.gate_in.highlight_frames > 0 else (0, 0, 255)
    col_out = (0, 255, 0) if state.gate_out.highlight_frames > 0 else (0, 0, 255)
    if state.gate_in.highlight_frames > 0:
        state.gate_in.highlight_frames -= 1
    if state.gate_out.highlight_frames > 0:
        state.gate_out.highlight_frames -= 1

    lin = state.gate_in.line_coords
    lout = state.gate_out.line_coords
    cv2.line(img, (lin[0], lin[1]), (lin[2], lin[3]), col_in, 3)
    cv2.putText(img, "GATE IN", (lin[0] + 10, lin[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_in, 2)

    cv2.line(img, (lout[0], lout[1]), (lout[2], lout[3]), col_out, 3)
    cv2.putText(img, "GATE OUT", (lout[0] + 10, lout[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_out, 2)

    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
        state.unique_ids.add(track_id)
        utils.draw_tracker_box(img, x1, y1, x2, y2, track_id, color=(255, 0, 255))

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (cx, cy), 4, (255, 0, 255), -1)

        trk = state.tracker.get_track_by_id(track_id)
        p_prev, p_curr = trk.get_last_two_centroids() if trk else ((cx, cy), (cx, cy))

        state.gate_in.check_crossing(track_id, p_prev, p_curr)
        state.gate_out.check_crossing(track_id, p_prev, p_curr)

    # Compute live FPS
    proc_time = time.time() - t_start
    live_fps = round(1.0 / proc_time, 1) if proc_time > 0 else 0.0

    in_cnt = state.gate_in.total_count
    out_cnt = state.gate_out.total_count
    occ_cnt = max(0, in_cnt - out_cnt)
    active_cnt = len(tracked_objects)

    # HUD overlay on webcam
    hud_text = f"IN: {in_cnt} | OUT: {out_cnt} | OCC: {occ_cnt} | LIVE: {active_cnt} | FPS: {live_fps}"
    cv2.rectangle(img, (10, 10), (w - 10, 45), (30, 30, 30), -1)
    cv2.putText(img, hud_text, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    annotated_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return annotated_rgb, in_cnt, out_cnt, occ_cnt, active_cnt, live_fps


def reset_webcam():
    global _WEBCAM_STATE
    _WEBCAM_STATE.reset()
    return "Webcam tracker and counters have been reset to zero."


def toggle_webcam(active_state):
    global _WEBCAM_STATE
    _WEBCAM_STATE.is_active = active_state
    return f"Webcam streaming is now {'Active' if active_state else 'Paused'}."


# ============================================================================
# 5. Gradio Dashboard UI Construction
# ============================================================================

CUSTOM_CSS = """
.metric-box {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.header-badge {
    background: #0284c7;
    color: white;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
}
"""

def create_dashboard():
    with gr.Blocks(title="AI People Counting & Flow Analytics") as demo:
        gr.Markdown(
            """
            # AI People Counting & Spatial Flow Analytics
            ### Production-grade Pedestrian Detection, Multi-Object Tracking & Flow Monitoring powered by **YOLOv8** & **SORT**
            """
        )

        with gr.Tabs():
            # ================================================================
            # TAB 1: Video Upload & Processing
            # ================================================================
            with gr.Tab("Video Analytics & Counting"):
                gr.Markdown("Upload an MP4 or AVI video to run full multi-object detection, tracking, directional counting, and flow analytics.")

                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="Upload Video (MP4 / AVI)", sources=["upload"])
                        
                        with gr.Row():
                            conf_slider = gr.Slider(minimum=0.1, maximum=0.9, value=0.3, step=0.05, label="Detection Confidence")
                            max_frames_input = gr.Number(value=0, label="Max Frames (0 = entire video)", precision=0)

                        mask_checkbox = gr.Checkbox(value=True, label="Apply Escalator ROI Mask (if available)")
                        process_btn = gr.Button("Analyze Video", variant="primary", size="lg")

                        # Sample Video Preset button
                        default_sample = utils.resolve_asset_path("people.mp4")
                        if os.path.exists(default_sample):
                            load_sample_btn = gr.Button("Load Default Sample (`people.mp4`)", variant="secondary")
                            load_sample_btn.click(lambda: default_sample, outputs=[video_input])

                    with gr.Column(scale=2):
                        output_video = gr.Video(label="Annotated Processed Video", interactive=False)

                gr.Markdown("### Real-Time Pipeline Metrics")
                with gr.Row():
                    kpi_in = gr.Number(label="Total IN (Upward)", value=0, precision=0)
                    kpi_out = gr.Number(label="Total OUT (Downward)", value=0, precision=0)
                    kpi_occ = gr.Number(label="Net Occupancy", value=0, precision=0)
                    kpi_unique = gr.Number(label="Total Unique IDs", value=0, precision=0)
                    kpi_fps = gr.Number(label="Processing FPS", value=0.0, precision=1)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Directional Balance (IN vs OUT)")
                        chart_in_out = gr.BarPlot(
                            x="Category", y="Count",
                            title="Total Flow Distribution",
                            y_lim=[0, None],
                            tooltip=["Category", "Count"],
                            height=300
                        )
                    with gr.Column():
                        gr.Markdown("#### Temporal Occupancy Curve")
                        chart_trend = gr.LinePlot(
                            x="Time (s)", y="Occupancy",
                            title="Occupancy Trend Across Video Duration",
                            tooltip=["Time (s)", "Occupancy"],
                            height=300
                        )

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("#### Granular Line-Crossing Event Audit Log")
                        events_table = gr.DataFrame(
                            headers=["Event ID", "Timestamp (s)", "Frame", "Track ID", "Direction", "Net Occupancy"],
                            interactive=False,
                            wrap=True
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("#### Session Audit & Export")
                        summary_box = gr.Markdown("Run a video to view summary statistics.")
                        download_csv_btn = gr.File(label="Download Verified Event Log (CSV)")

                process_btn.click(
                    fn=process_video,
                    inputs=[video_input, conf_slider, max_frames_input, mask_checkbox],
                    outputs=[
                        output_video,
                        kpi_in,
                        kpi_out,
                        kpi_occ,
                        kpi_unique,
                        kpi_fps,
                        chart_in_out,
                        chart_trend,
                        events_table,
                        download_csv_btn,
                        summary_box
                    ]
                )

            # ================================================================
            # TAB 2: Image Detection
            # ================================================================
            with gr.Tab("Static Image Detection"):
                gr.Markdown("Upload any static image to detect pedestrians, count total individuals, and inspect detection confidence scores.")
                with gr.Row():
                    with gr.Column():
                        img_input = gr.Image(type="numpy", label="Upload Image")
                        img_conf = gr.Slider(minimum=0.1, maximum=0.9, value=0.3, step=0.05, label="Detection Confidence")
                        detect_btn = gr.Button("Detect People", variant="primary", size="lg")
                    with gr.Column():
                        img_output = gr.Image(type="numpy", label="Detection Annotations")
                        img_count_kpi = gr.Number(label="Total People Detected", value=0, precision=0)
                        img_stats = gr.Markdown("Upload an image and click Detect.")

                gr.Markdown("#### Detection Breakdown")
                detections_table = gr.DataFrame(
                    headers=["Person #", "Confidence", "Bounding Box", "Width (px)", "Height (px)"],
                    interactive=False
                )

                detect_btn.click(
                    fn=process_image,
                    inputs=[img_input, img_conf],
                    outputs=[img_output, img_count_kpi, img_stats, detections_table]
                )

            # ================================================================
            # TAB 3: Live Webcam Streaming
            # ================================================================
            with gr.Tab("Live Webcam Monitoring"):
                gr.Markdown("Real-time webcam inference with persistent SORT tracking and multi-gate directional counting.")
                with gr.Row():
                    with gr.Column(scale=2):
                        webcam_input = gr.Image(sources=["webcam"], streaming=True, label="Live Webcam Feed")
                        with gr.Row():
                            stream_conf = gr.Slider(minimum=0.1, maximum=0.9, value=0.3, step=0.05, label="Confidence Threshold")
                            toggle_btn = gr.Checkbox(value=True, label="Streaming Active")
                            reset_webcam_btn = gr.Button("Reset Counters & Tracks", variant="secondary")

                        webcam_status = gr.Markdown("Live streaming ready. Enable your browser camera above.")

                    with gr.Column(scale=1):
                        live_in = gr.Number(label="Live Total IN", value=0, precision=0)
                        live_out = gr.Number(label="Live Total OUT", value=0, precision=0)
                        live_occ = gr.Number(label="Current Occupancy", value=0, precision=0)
                        live_active = gr.Number(label="People Currently in View", value=0, precision=0)
                        live_fps = gr.Number(label="Processing FPS", value=0.0, precision=1)

                webcam_input.stream(
                    fn=process_webcam_frame,
                    inputs=[webcam_input, stream_conf],
                    outputs=[webcam_input, live_in, live_out, live_occ, live_active, live_fps]
                )

                reset_webcam_btn.click(
                    fn=reset_webcam,
                    inputs=[],
                    outputs=[webcam_status]
                )

                toggle_btn.change(
                    fn=toggle_webcam,
                    inputs=[toggle_btn],
                    outputs=[webcam_status]
                )

    return demo


if __name__ == "__main__":
    demo = create_dashboard()
    demo.queue()
    print("Launching People Counting Dashboard on http://127.0.0.1:7860 ...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS
    )
