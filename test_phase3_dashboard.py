"""
Phase 3 Dashboard Integration Test Suite.
Tests all dashboard backend functions, state persistence, error handling,
video/image processing, CSV generation, and Gradio Blocks construction.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd

from app import (
    get_yolo_model,
    process_video,
    process_image,
    process_webcam_frame,
    reset_webcam,
    toggle_webcam,
    create_dashboard,
    _WEBCAM_STATE
)
import utils


def run_tests():
    print("=" * 70)
    print("PHASE 3 DASHBOARD INTEGRATION TEST SUITE")
    print("=" * 70)

    results = {}

    # Test 1: Model Singleton
    print("\n--- Test 1: YOLO Model Singleton Lifecycle ---")
    m1 = get_yolo_model()
    m2 = get_yolo_model()
    if m1 is m2:
        print("PASS: get_yolo_model() returned identical singleton instance.")
        results["Model Singleton"] = "PASS"
    else:
        print("FAIL: get_yolo_model() created multiple instances.")
        results["Model Singleton"] = "FAIL"

    # Test 2: Error Handling for Invalid Inputs
    print("\n--- Test 2: Invalid Inputs & Error Handling ---")
    try:
        vid_err = process_video("non_existent_file.mp4", 0.3, 10, False)
        img_err = process_image(None, 0.3)
        cam_err = process_webcam_frame(None, 0.3)

        assert vid_err[0] is None, "process_video should return None for video path on invalid file"
        assert "Error" in vid_err[-1], "process_video summary should contain error message"
        assert img_err[0] is None and img_err[1] == 0, "process_image should handle None gracefully"
        assert cam_err[0] is None and cam_err[1] == 0, "process_webcam_frame should handle None gracefully"

        print("PASS: Gracefully handled invalid inputs without unhandled exceptions.")
        results["Error Handling"] = "PASS"
    except Exception as e:
        print(f"FAIL: Error handling raised exception: {e}")
        results["Error Handling"] = f"FAIL ({e})"

    # Test 3: Static Image Detection Pipeline
    print("\n--- Test 3: Static Image Detection Pipeline ---")
    try:
        sample_img_path = utils.resolve_asset_path("snapshot_frame_100.png")
        if not os.path.exists(sample_img_path):
            # Create a test frame from people.mp4 if snapshot not found
            cap = cv2.VideoCapture(utils.resolve_asset_path("people.mp4"))
            ret, frame = cap.read()
            cap.release()
            cv2.imwrite("test_sample_frame.png", frame)
            sample_img_path = "test_sample_frame.png"

        bgr = cv2.imread(sample_img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        annotated_rgb, count, stats, df = process_image(rgb, conf_threshold=0.3)

        assert annotated_rgb is not None, "Annotated image is None"
        assert annotated_rgb.shape == rgb.shape, f"Shape mismatch: {annotated_rgb.shape} vs {rgb.shape}"
        assert isinstance(count, int), f"Count is not int: {type(count)}"
        assert isinstance(df, pd.DataFrame), "Detection table is not a DataFrame"
        assert count == len(df), f"Count ({count}) does not match DataFrame rows ({len(df)})"

        print(f"PASS: Detected {count} people in static image. Stats: {stats}")
        results["Static Image Detection"] = f"PASS ({count} people detected)"
    except Exception as e:
        print(f"FAIL: Static Image Detection failed: {e}")
        results["Static Image Detection"] = f"FAIL ({e})"

    # Test 4: Video Upload & Processing Pipeline
    print("\n--- Test 4: Video Upload & Processing Pipeline (people.mp4, 50 frames) ---")
    try:
        sample_video_path = utils.resolve_asset_path("people.mp4")
        # Run for 50 frames to verify end-to-end processing quickly
        ret_vals = process_video(
            video_path=sample_video_path,
            conf_threshold=0.3,
            max_frames=50,
            apply_mask=True
        )

        (out_video_path, total_in, total_out, net_occ, unique_count,
         avg_fps, in_out_df, trend_df, events_df, csv_path, summary_text) = ret_vals

        assert out_video_path is not None and os.path.exists(out_video_path), f"Output video missing: {out_video_path}"
        assert os.path.getsize(out_video_path) > 0, "Output video has size 0"
        assert csv_path is not None and os.path.exists(csv_path), f"CSV report missing: {csv_path}"
        assert os.path.getsize(csv_path) > 0, "CSV report has size 0"
        assert isinstance(in_out_df, pd.DataFrame), "in_out_df is not DataFrame"
        assert isinstance(trend_df, pd.DataFrame), "trend_df is not DataFrame"
        assert isinstance(events_df, pd.DataFrame), "events_df is not DataFrame"
        assert avg_fps > 0, f"Average FPS is non-positive: {avg_fps}"

        print(f"PASS: Video processed successfully.")
        print(f"      Output Video: {out_video_path} ({os.path.getsize(out_video_path)} bytes)")
        print(f"      Metrics: IN={total_in}, OUT={total_out}, Net Occ={net_occ}, Unique IDs={unique_count}, FPS={avg_fps}")
        print(f"      CSV Report: {csv_path} ({os.path.getsize(csv_path)} bytes)")
        results["Video Processing"] = f"PASS ({avg_fps} FPS, IN={total_in}, OUT={total_out})"
    except Exception as e:
        import traceback
        traceback.print_exc()
        results["Video Processing"] = f"FAIL ({e})"

    # Test 5: Live Webcam Streaming & State Persistence
    print("\n--- Test 5: Live Webcam Streaming & State Persistence ---")
    try:
        reset_msg = reset_webcam()
        assert _WEBCAM_STATE.frame_idx == 0, "Webcam state not reset"

        # Read 10 frames from video as simulated webcam frames
        cap = cv2.VideoCapture(utils.resolve_asset_path("people.mp4"))
        webcam_results = []
        for i in range(10):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame, in_c, out_c, occ_c, active_c, live_fps = process_webcam_frame(frame_rgb, conf_threshold=0.3)
            webcam_results.append((annotated_frame, in_c, out_c, occ_c, active_c, live_fps))
        cap.release()

        assert len(webcam_results) == 10, "Failed to process 10 simulated webcam frames"
        assert _WEBCAM_STATE.frame_idx == 10, f"Expected frame_idx 10, got {_WEBCAM_STATE.frame_idx}"
        assert len(_WEBCAM_STATE.unique_ids) > 0, "Webcam tracker failed to accumulate unique IDs"

        # Test Pause Toggle
        pause_msg = toggle_webcam(False)
        assert _WEBCAM_STATE.is_active is False, "Webcam toggle pause failed"
        # When paused, process_webcam_frame should return input without processing
        p_frame, _, _, _, _, _ = process_webcam_frame(frame_rgb, 0.3)
        assert np.array_equal(p_frame, frame_rgb), "Paused webcam did not pass-through frame"

        # Resume
        toggle_webcam(True)
        assert _WEBCAM_STATE.is_active is True, "Webcam toggle resume failed"

        # Reset
        reset_webcam()
        assert _WEBCAM_STATE.frame_idx == 0 and len(_WEBCAM_STATE.unique_ids) == 0, "Reset failed to clear state"

        print(f"PASS: Webcam streaming preserved state across frames, paused/resumed, and reset cleanly.")
        results["Webcam Streaming & State"] = "PASS"
    except Exception as e:
        import traceback
        traceback.print_exc()
        results["Webcam Streaming & State"] = f"FAIL ({e})"

    # Test 6: Gradio Blocks Construction
    print("\n--- Test 6: Gradio Blocks UI Hierarchy Construction ---")
    try:
        demo = create_dashboard()
        assert demo is not None, "create_dashboard() returned None"
        # Check that blocks contains components
        num_blocks = len(demo.blocks)
        print(f"PASS: Gradio Blocks initialized successfully with {num_blocks} components.")
        results["Gradio Blocks Construction"] = f"PASS ({num_blocks} components)"
    except Exception as e:
        import traceback
        traceback.print_exc()
        results["Gradio Blocks Construction"] = f"FAIL ({e})"

    # Test 7: CSV Report Format Verification
    print("\n--- Test 7: CSV Report Format & Header Verification ---")
    try:
        reports = list(Path("reports").glob("*.csv"))
        assert len(reports) > 0, "No CSV reports found in reports/ directory"
        latest_report = sorted(reports, key=lambda p: p.stat().st_mtime)[-1]
        df_csv = pd.read_csv(latest_report)
        expected_cols = ["Event ID", "Timestamp (s)", "Frame", "Track ID", "Direction", "Net Occupancy"]
        assert list(df_csv.columns) == expected_cols, f"CSV headers mismatch: {list(df_csv.columns)}"
        print(f"PASS: Verified latest CSV report '{latest_report.name}' with {len(df_csv)} events and expected schema.")
        results["CSV Report Schema"] = f"PASS ({latest_report.name})"
    except Exception as e:
        import traceback
        traceback.print_exc()
        results["CSV Report Schema"] = f"FAIL ({e})"

    print("\n" + "=" * 70)
    print("PHASE 3 TEST SUMMARY")
    print("=" * 70)
    all_passed = True
    for test_name, status in results.items():
        print(f"[{'PASS' if 'PASS' in status else 'FAIL'}] {test_name}: {status}")
        if "FAIL" in status:
            all_passed = False

    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    from pathlib import Path
    success = run_tests()
    sys.exit(0 if success else 1)

