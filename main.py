"""
People Counting using YOLOv8, SORT Tracking, and OpenCV.
Counts people moving up and down escalators using robust line-crossing trajectory mathematics.
"""

import sys
import os
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from sort import Sort
import utils


def parse_args():
    parser = argparse.ArgumentParser(description="People Counting using YOLOv8 & SORT")
    parser.add_argument("--source", type=str, default="people.mp4",
                        help="Video file path or webcam index (default: people.mp4)")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLO model path (default: yolov8n.pt)")
    parser.add_argument("--mask", type=str, default="mask-1.png",
                        help="ROI mask path (default: mask-1.png)")
    parser.add_argument("--graphics", type=str, default="graphics-1.png",
                        help="Overlay graphics dashboard path (default: graphics-1.png)")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Person detection confidence threshold (default: 0.3)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI display (for background processing or tests)")
    parser.add_argument("--save-out", type=str, default="",
                        help="Save processed output video to the specified path")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Maximum frames to process (0 = process entire video)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Resolve asset paths independently of working directory
    source_path = utils.resolve_asset_path(args.source)
    model_path = utils.resolve_asset_path(args.model)
    mask_path = utils.resolve_asset_path(args.mask)
    graphics_path = utils.resolve_asset_path(args.graphics)

    print(f"=== People Counting System Initializing ===")
    print(f"Source video: {source_path}")
    print(f"YOLO Model:   {model_path}")
    print(f"Mask ROI:     {mask_path}")
    print(f"Dashboard UI: {graphics_path}")

    # 2. Open video source
    is_webcam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_webcam else source_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video source '{source_path}'", file=sys.stderr)
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Stream: {width}x{height} @ {fps:.1f} FPS, {total_frames} total frames")

    # 3. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference device: {device}")
    model = YOLO(model_path)

    # 4. Load static assets ONCE outside processing loop (Eliminates disk I/O bottleneck)
    mask = None
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path)
        if mask is not None and (mask.shape[1] != width or mask.shape[0] != height):
            mask = cv2.resize(mask, (width, height))
        print("ROI Mask loaded successfully.")
    else:
        print("WARNING: ROI Mask not found. Processing entire frame without masking.")

    img_graphics = None
    if os.path.exists(graphics_path):
        img_graphics = utils.load_overlay_image(graphics_path)
        print("Dashboard graphics loaded successfully.")
    else:
        print("WARNING: Dashboard graphics not found. Using text overlay.")

    # 5. Initialize SORT Tracker
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    tracker.reset()

    # 6. Stateful Line Gates configuration
    # Escalator UP (left side: horizontal line at y=161, moving upward dy < 0)
    limits_up = [103, 161, 296, 161]
    gate_up = utils.LineGate("UP", limits_up, direction='up', hysteresis=15)

    # Escalator DOWN (right side: horizontal line at y=489, moving downward dy > 0)
    limits_down = [527, 489, 735, 489]
    gate_down = utils.LineGate("DOWN", limits_down, direction='down', hysteresis=15)

    # Optional video writer
    writer = None
    if args.save_out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_out, fourcc, fps, (width, height))
        print(f"Recording output to: {args.save_out}")

    frame_idx = 0
    start_time = time.time()
    print("Beginning video processing loop...")

    while True:
        success, img = cap.read()
        
        # Safe EOF handling: gracefully exit when video ends or frame fails
        if not success or img is None:
            print(f"\nEnd of video stream reached at frame {frame_idx}.")
            break

        frame_idx += 1
        if args.max_frames > 0 and frame_idx > args.max_frames:
            print(f"\nReached max frames limit ({args.max_frames}). Stopping.")
            break

        # Apply ROI mask if available
        if mask is not None:
            img_region = cv2.bitwise_and(img, mask)
        else:
            img_region = img

        # Overlay dashboard graphics
        if img_graphics is not None:
            img = utils.overlay_transparent(img, img_graphics, 730, 260)

        # YOLO detection (class 0 = person)
        results = model.predict(img_region, classes=[0], conf=args.conf, verbose=False, device=device)

        detections = np.empty((0, 5))
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                detections = np.vstack((detections, [int(x1), int(y1), int(x2), int(y2), conf]))

        # Update SORT tracker
        tracked_objects = tracker.update(detections)

        # Draw counting lines with active highlight state
        color_up = (0, 255, 0) if gate_up.highlight_frames > 0 else (0, 0, 255)
        color_down = (0, 255, 0) if gate_down.highlight_frames > 0 else (0, 0, 255)
        if gate_up.highlight_frames > 0:
            gate_up.highlight_frames -= 1
        if gate_down.highlight_frames > 0:
            gate_down.highlight_frames -= 1

        cv2.line(img, (limits_up[0], limits_up[1]), (limits_up[2], limits_up[3]), color_up, 5)
        cv2.line(img, (limits_down[0], limits_down[1]), (limits_down[2], limits_down[3]), color_down, 5)

        # Process each tracked person
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj
            x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
            w, h = x2 - x1, y2 - y1

            # Draw bounding box and ID tag
            utils.draw_tracker_box(img, x1, y1, x2, y2, track_id, color=(255, 0, 255))

            # Current centroid
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Retrieve trajectory history from tracker
            trk = tracker.get_track_by_id(track_id)
            if trk is not None:
                p_prev, p_curr = trk.get_last_two_centroids()
            else:
                p_prev, p_curr = (cx, cy), (cx, cy)

            # Check crossings via stateful LineGates
            if gate_up.check_crossing(track_id, p_prev, p_curr):
                cv2.line(img, (limits_up[0], limits_up[1]), (limits_up[2], limits_up[3]), (0, 255, 0), 5)

            if gate_down.check_crossing(track_id, p_prev, p_curr):
                cv2.line(img, (limits_down[0], limits_down[1]), (limits_down[2], limits_down[3]), (0, 255, 0), 5)

        # Render counter numbers onto dashboard
        count_up_str = str(gate_up.total_count)
        count_down_str = str(gate_down.total_count)
        cv2.putText(img, count_up_str, (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
        cv2.putText(img, count_down_str, (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)

        # Write frame to output video if requested
        if writer is not None:
            writer.write(img)

        # GUI display if not in headless mode
        if not args.headless:
            cv2.imshow("People Counting - OpenCV & YOLOv8", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nUser requested exit ('q' pressed).")
                break

        # Progress reporting every 60 frames
        if frame_idx % 60 == 0:
            elapsed = time.time() - start_time
            current_fps = frame_idx / elapsed if elapsed > 0 else 0
            print(f"Frame {frame_idx}/{total_frames} ({frame_idx*100//total_frames}%) - "
                  f"UP: {count_up_str}, DOWN: {count_down_str} - FPS: {current_fps:.1f}")

    # Cleanup resources
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0
    print("\n=== Processing Complete ===")
    print(f"Total Frames Processed: {frame_idx}")
    print(f"Total Time:             {total_time:.2f} seconds ({avg_fps:.1f} FPS)")
    print(f"Final Count UP:         {gate_up.total_count}")
    print(f"Final Count DOWN:       {gate_down.total_count}")
    print(f"Total People Counted:   {gate_up.total_count + gate_down.total_count}")


if __name__ == "__main__":
    main()
