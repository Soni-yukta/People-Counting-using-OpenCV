"""
Targeted test verifying video line crossing event capture,
charts data integrity, CSV download content, and metrics accuracy.
"""

import os
import sys
import pandas as pd
from app import process_video
import utils

def test_events_and_charts():
    print("Testing Video Processing with Event Logging & Chart Generation...")
    video_path = utils.resolve_asset_path("people.mp4")
    assert os.path.exists(video_path), f"Sample video not found at {video_path}"

    # Run for 220 frames where people cross both lines
    (out_vid, total_in, total_out, occ, unique_ids, fps,
     in_out_df, trend_df, events_df, csv_path, summary) = process_video(
        video_path=video_path,
        conf_threshold=0.3,
        max_frames=220,
        apply_mask=True
    )

    print(f"Processed 220 frames:")
    print(f"  Total IN: {total_in}")
    print(f"  Total OUT: {total_out}")
    print(f"  Net Occupancy: {occ}")
    print(f"  Unique Tracks: {unique_ids}")
    print(f"  Processing FPS: {fps}")
    print(f"  Events recorded: {len(events_df)}")
    print(f"  Trend data points: {len(trend_df)}")

    # Assertions
    assert os.path.exists(out_vid) and os.path.getsize(out_vid) > 0, "Output video missing or empty"
    assert os.path.exists(csv_path) and os.path.getsize(csv_path) > 0, "CSV file missing or empty"
    assert total_in >= 1, f"Expected at least 1 IN crossing by frame 220, got {total_in}"
    assert len(events_df) >= 1, f"Expected at least 1 event in events_df, got {len(events_df)}"
    assert not in_out_df.empty, "in_out_df is empty"
    assert not trend_df.empty, "trend_df is empty"

    # Verify CSV file contents match DataFrame
    csv_loaded = pd.read_csv(csv_path)
    assert len(csv_loaded) == len(events_df), f"CSV row count {len(csv_loaded)} != events_df rows {len(events_df)}"
    print(f"Events table sample:\n{csv_loaded.to_string()}")

    print("\nALL EVENT AND CHART TESTS PASSED SUCCESSFULLY!")
    return True

if __name__ == "__main__":
    success = test_events_and_charts()
    sys.exit(0 if success else 1)

