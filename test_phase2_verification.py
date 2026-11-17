"""
Independent Verification Test Suite for Phase 2: Core Computer Vision.
Tests:
- Test 3: Stationary people rejection
- Test 4: Wrong-direction rejection
- Test 5: One person crossing upward then downward
- Test 6: Multiple people crossing simultaneously
- Test 7: Temporary tracking loss recovery
- Test 8: Legitimate repeated crossings handling
- Test 9: Tracker ID reset between separate video sessions
"""

import sys
import numpy as np
import sort
import utils


def test_stationary_person():
    """Verify that stationary people on or near the line do not get counted."""
    gate = utils.LineGate("UP", [100, 161, 300, 161], direction='up')
    
    # Exactly on line, 0 movement
    for _ in range(30):
        c = gate.check_crossing(1, (200, 161), (200, 161))
        if c:
            return False, "Counted completely stationary person at (200, 161)"

    # Stationary with sub-pixel Kalman jitter (+- 0.5px) around y=161
    np.random.seed(42)
    for _ in range(30):
        prev_pt = (200 + np.random.uniform(-0.3, 0.3), 161 + np.random.uniform(-0.5, 0.5))
        curr_pt = (200 + np.random.uniform(-0.3, 0.3), 161 + np.random.uniform(-0.5, 0.5))
        c = gate.check_crossing(2, prev_pt, curr_pt)
        if c:
            return False, "Counted jittering stationary person near y=161"

    if gate.total_count != 0:
        return False, f"Expected 0 counts, got {gate.total_count}"
    return True, "Stationary person and sub-pixel jitter were correctly rejected (0 counts)"


def test_wrong_direction():
    """Verify that people moving in the wrong direction are rejected."""
    gate_up = utils.LineGate("UP", [100, 161, 300, 161], direction='up')
    gate_down = utils.LineGate("DOWN", [500, 489, 700, 489], direction='down')

    # Person moving DOWN on UP line (y from 140 to 180, dy = +40)
    c1 = gate_up.check_crossing(1, (200, 140), (200, 180))
    if c1 or gate_up.total_count != 0:
        return False, f"Wrong direction counted on UP line! Count: {gate_up.total_count}"

    # Person moving UP on DOWN line (y from 510 to 460, dy = -50)
    c2 = gate_down.check_crossing(2, (600, 510), (600, 460))
    if c2 or gate_down.total_count != 0:
        return False, f"Wrong direction counted on DOWN line! Count: {gate_down.total_count}"

    return True, "Wrong-direction traversals were correctly rejected on both gates (0 counts)"


def test_upward_then_downward():
    """Test one person crossing upward through gate UP, then downward through gate DOWN."""
    gate_up = utils.LineGate("UP", [100, 161, 300, 161], direction='up')
    gate_down = utils.LineGate("DOWN", [500, 489, 700, 489], direction='down')

    # Person 1 walks UP escalator: y: 190 -> 175 -> 145 (crosses 161)
    gate_up.check_crossing(1, (200, 190), (200, 175))
    crossed_up = gate_up.check_crossing(1, (200, 175), (200, 145))

    if not crossed_up or gate_up.total_count != 1:
        return False, f"Expected UP count = 1, got {gate_up.total_count}"

    # Person 1 walks across the floor, then walks DOWN escalator: y: 460 -> 475 -> 515 (crosses 489)
    gate_down.check_crossing(1, (600, 460), (600, 475))
    crossed_down = gate_down.check_crossing(1, (600, 475), (600, 515))

    if not crossed_down or gate_down.total_count != 1:
        return False, f"Expected DOWN count = 1, got {gate_down.total_count}"

    return True, f"Person 1 successfully counted on UP gate ({gate_up.total_count}) and DOWN gate ({gate_down.total_count})"


def test_simultaneous_crossings():
    """Test multiple people crossing the line at the exact same frame."""
    gate = utils.LineGate("UP", [100, 161, 300, 161], direction='up')

    # Person 1 at x=150 crosses y=161
    c1 = gate.check_crossing(1, (150, 180), (150, 140))
    # Person 2 at x=250 crosses y=161 in same frame
    c2 = gate.check_crossing(2, (250, 185), (250, 145))
    # Person 3 at x=200 is still below line
    c3 = gate.check_crossing(3, (200, 220), (200, 190))

    if not c1 or not c2 or c3:
        return False, f"Simultaneous detection mismatch: c1={c1}, c2={c2}, c3={c3}"
    if gate.total_count != 2:
        return False, f"Expected total_count = 2, got {gate.total_count}"

    return True, f"Multiple people crossing simultaneously counted accurately (count = {gate.total_count})"


def test_temporary_tracking_loss():
    """Test temporary occlusion/tracking drop using SORT."""
    tracker = sort.Sort(max_age=10, min_hits=2, iou_threshold=0.3)
    tracker.reset()
    gate = utils.LineGate("UP", [100, 161, 300, 161], direction='up')

    # Frame 0: detection below line
    det0 = np.array([[150, 200, 190, 250, 0.9]])
    tracker.update(det0)

    # Frame 1: moving upward
    det1 = np.array([[150, 180, 190, 230, 0.9]])
    res1 = tracker.update(det1)
    track_id = int(res1[0, 4])

    # Frames 2, 3, 4: OCCLUSION / DETECTION LOSS (empty detections)
    # SORT Kalman filter should predict positions
    tracker.update(np.empty((0, 5)))
    tracker.update(np.empty((0, 5)))
    tracker.update(np.empty((0, 5)))

    # Frame 5: Person re-emerges above line at y=140
    det5 = np.array([[150, 120, 190, 170, 0.9]])
    res5 = tracker.update(det5)

    if len(res5) == 0:
        return False, "Tracker failed to re-acquire object after 3 frames of occlusion"

    reacquired_id = int(res5[0, 4])
    if reacquired_id != track_id:
        return False, f"Tracker assigned new ID {reacquired_id} instead of preserving {track_id}"

    trk = tracker.get_track_by_id(reacquired_id)
    p_prev, p_curr = trk.get_last_two_centroids()
    crossed = gate.check_crossing(reacquired_id, p_prev, p_curr)

    if not crossed or gate.total_count != 1:
        return False, f"Line crossing missed during occlusion recovery (count: {gate.total_count})"

    return True, f"Track ID {track_id} preserved across 3-frame occlusion and crossing registered (count = 1)"


def test_repeated_crossings():
    """Test that legitimate repeated crossings are counted while immediate duplicates are prevented."""
    gate = utils.LineGate("UP", [100, 161, 300, 161], direction='up', hysteresis=15)

    # Pass 1: Person 1 walks UP (180 -> 140)
    c1 = gate.check_crossing(1, (180, 180), (180, 140))
    if not c1 or gate.total_count != 1:
        return False, f"Pass 1 failed: c1={c1}, count={gate.total_count}"

    # Jittering above the line should NOT count again
    for _ in range(10):
        c = gate.check_crossing(1, (180, 140), (180, 138))
        if c:
            return False, "Duplicate count triggered while staying on exit side!"

    # Person 1 walks back down to below line (138 -> 190)
    # This should re-arm the gate because cy = 190 > 161 + 15
    gate.check_crossing(1, (180, 138), (180, 190))

    # Pass 2: Person 1 walks UP AGAIN (190 -> 145)
    c2 = gate.check_crossing(1, (180, 190), (180, 145))
    if not c2 or gate.total_count != 2:
        return False, f"Legitimate Pass 2 failed: c2={c2}, count={gate.total_count}"

    return True, f"Legitimate repeated crossings counted correctly without duplicate jitter (count = {gate.total_count})"


def test_tracker_id_reset():
    """Verify that tracker IDs reset to 1 between separate sessions."""
    tracker = sort.Sort()
    tracker.reset()

    # Session 1
    for f in range(5):
        tracker.update(np.array([[100, 100, 150, 150, 0.9]]))
    tracks_session_1 = tracker.update(np.array([[100, 100, 150, 150, 0.9]]))
    id_session_1 = int(tracks_session_1[0, 4])

    # Reset for Session 2
    tracker.reset()
    for f in range(5):
        tracker.update(np.array([[200, 200, 250, 250, 0.9]]))
    tracks_session_2 = tracker.update(np.array([[200, 200, 250, 250, 0.9]]))
    id_session_2 = int(tracks_session_2[0, 4])

    if id_session_1 != 1 or id_session_2 != 1:
        return False, f"ID reset failed: session 1 start ID={id_session_1}, session 2 start ID={id_session_2}"

    return True, f"Tracker ID cleanly resets to 1 between sessions (Session 1: {id_session_1}, Session 2: {id_session_2})"


def run_all_tests():
    tests = [
        ("Test 3: Stationary Person Rejection", test_stationary_person),
        ("Test 4: Wrong Direction Rejection", test_wrong_direction),
        ("Test 5: Upward Then Downward Crossings", test_upward_then_downward),
        ("Test 6: Simultaneous Multi-Person Crossings", test_simultaneous_crossings),
        ("Test 7: Temporary Tracking Loss & Occlusion", test_temporary_tracking_loss),
        ("Test 8: Legitimate Repeated Crossings", test_repeated_crossings),
        ("Test 9: Tracker ID Reset Between Sessions", test_tracker_id_reset),
    ]

    print("=== RUNNING PHASE 2 INDEPENDENT VERIFICATION TESTS ===")
    all_passed = True
    for name, test_fn in tests:
        try:
            passed, msg = test_fn()
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] {name}")
            print(f"       Details: {msg}")
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"[FAIL] {name} - EXCEPTION: {e}")
            all_passed = False

    print("=====================================================")
    if all_passed:
        print("ALL TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("ONE OR MORE TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()

