"""
Utility functions and classes for People Counting using OpenCV.
Provides robust geometry (line segment crossing), path resolution, stateful LineGate,
and OpenCV drawing helpers.
"""

import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image


def resolve_asset_path(filename):
    """
    Finds the absolute path to an asset, searching current dir,
    the script's directory, and the nested subfolder.
    """
    search_dirs = [
        Path.cwd(),
        Path.cwd() / "People-Counting-using-OpenCV-main",
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent / "People-Counting-using-OpenCV-main",
        Path(__file__).resolve().parent.parent,
    ]
    for d in search_dirs:
        candidate = d / filename
        if candidate.exists():
            return str(candidate.resolve())
    return str(Path(filename).resolve())


def ccw(A, B, C):
    """Checks whether the turn formed by points A, B, C is counterclockwise."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(A, B, C, D):
    """
    Checks if line segment AB intersects with line segment CD using the orientation test.
    A, B: (x, y) endpoints of segment 1 (e.g., previous and current object centroid).
    C, D: (x, y) endpoints of segment 2 (e.g., counting line).
    """
    if A is None or B is None or C is None or D is None:
        return False

    # Bounding box quick rejection
    if (min(A[0], B[0]) > max(C[0], D[0]) or max(A[0], B[0]) < min(C[0], D[0]) or
            min(A[1], B[1]) > max(C[1], D[1]) or max(A[1], B[1]) < min(C[1], D[1])):
        return False

    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))


def is_crossing_line(prev_centroid, curr_centroid, line_coords, direction_filter=None, min_movement=1.5):
    """
    Determines if an object crossed a counting line between previous and current frames.
    
    line_coords: [x1, y1, x2, y2]
    direction_filter: 
      - 'up': requires dy = curr_y - prev_y < 0 (moving upward)
      - 'down': requires dy = curr_y - prev_y > 0 (moving downward)
      - None: any direction
    min_movement: minimum movement required to filter out stationary jitter
    """
    if prev_centroid is None or curr_centroid is None:
        return False

    px, py = float(prev_centroid[0]), float(prev_centroid[1])
    cx, cy = float(curr_centroid[0]), float(curr_centroid[1])

    # 1. Stationary filter
    dy = cy - py
    dx = cx - px
    displacement = np.sqrt(dx * dx + dy * dy)
    if displacement < min_movement:
        return False

    # 2. Direction filter
    if direction_filter == 'up' and dy > -min_movement:
        return False
    if direction_filter == 'down' and dy < min_movement:
        return False

    # 3. Geometric line segment intersection
    x1_line, y1_line, x2_line, y2_line = line_coords
    A = (px, py)
    B = (cx, cy)
    C = (float(x1_line), float(y1_line))
    D = (float(x2_line), float(y2_line))

    if segments_intersect(A, B, C, D):
        return True

    # 4. Straddle check for horizontal lines
    line_y = (y1_line + y2_line) / 2.0
    min_x = min(x1_line, x2_line) - 10
    max_x = max(x1_line, x2_line) + 10

    if abs(dy) > 1e-5:
        x_cross = px + (line_y - py) * (dx / dy)
    else:
        x_cross = (px + cx) / 2.0

    if min_x <= x_cross <= max_x:
        if (py >= line_y and cy <= line_y) or (py <= line_y and cy >= line_y):
            return True

    return False


class LineGate:
    """
    Stateful directional counting gate.
    Prevents duplicate counts and stationary false positives,
    while correctly allowing legitimate repeated crossings when an object
    leaves and re-approaches the line.
    """
    def __init__(self, name, line_coords, direction='up', hysteresis=15, min_movement=1.5):
        self.name = name
        self.line_coords = line_coords
        self.direction = direction.lower()
        self.hysteresis = hysteresis
        self.min_movement = min_movement

        self.total_count = 0
        self.counted_ids = set()
        self.track_states = {}  # track_id -> 'armed' or 'counted'
        self.highlight_frames = 0

    def reset(self):
        """Resets gate count and tracking states."""
        self.total_count = 0
        self.counted_ids.clear()
        self.track_states.clear()
        self.highlight_frames = 0

    def check_crossing(self, track_id, prev_centroid, curr_centroid):
        """
        Evaluates whether track_id crossed the gate.
        Returns True if a valid crossing occurred, False otherwise.
        """
        if prev_centroid is None or curr_centroid is None:
            return False

        px, py = float(prev_centroid[0]), float(prev_centroid[1])
        cx, cy = float(curr_centroid[0]), float(curr_centroid[1])

        # 1. Stationary check: filter out jitter
        dy = cy - py
        dx = cx - px
        displacement = np.sqrt(dx * dx + dy * dy)
        if displacement < self.min_movement:
            return False

        line_y = (self.line_coords[1] + self.line_coords[3]) / 2.0

        # 2. Direction check & Re-arming logic for legitimate repeated crossings
        if self.direction == 'up':
            if dy > -self.min_movement:
                # Moving downward or stationary; check if re-arming is warranted
                if self.track_states.get(track_id) == 'counted':
                    if cy > line_y + self.hysteresis:
                        # Object moved back below the line! Re-arm for legitimate subsequent crossing
                        self.track_states[track_id] = 'armed'
                return False

        elif self.direction == 'down':
            if dy < self.min_movement:
                # Moving upward or stationary; check if re-arming is warranted
                if self.track_states.get(track_id) == 'counted':
                    if cy < line_y - self.hysteresis:
                        # Object moved back above the line! Re-arm for legitimate subsequent crossing
                        self.track_states[track_id] = 'armed'
                return False

        # 3. Check if track is already counted in this pass
        state = self.track_states.get(track_id, 'armed')
        if state == 'counted':
            return False

        # 4. Check geometric intersection
        crossed = is_crossing_line(
            (px, py), (cx, cy),
            self.line_coords,
            direction_filter=self.direction,
            min_movement=self.min_movement
        )

        if crossed:
            self.total_count += 1
            self.track_states[track_id] = 'counted'
            self.counted_ids.add(track_id)
            self.highlight_frames = 15
            return True

        return False


def load_overlay_image(filepath):
    """
    Loads an image from filepath and returns it as a BGRA NumPy array.
    Supports RGBA and Palette (mode P) PNGs.
    """
    if not os.path.exists(filepath):
        return None
    pil_img = Image.open(filepath).convert('RGBA')
    bgra = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
    return bgra


def overlay_transparent(background, overlay, x, y):
    """
    Overlays a BGRA image onto a BGR background image with proper alpha blending.
    Handles out-of-bounds positioning safely with clipping.
    """
    if overlay is None or background is None:
        return background

    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGRA)
    elif overlay.shape[2] == 3:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

    h, w = overlay.shape[:2]
    bg_h, bg_w = background.shape[:2]

    if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0:
        return background

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + w)
    y2 = min(bg_h, y + h)

    overlay_x1 = max(0, -x)
    overlay_y1 = max(0, -y)
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    overlay_crop = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    bg_crop = background[y1:y2, x1:x2]

    alpha = (overlay_crop[:, :, 3] / 255.0)[:, :, np.newaxis]
    color = overlay_crop[:, :, :3]

    blended = (alpha * color + (1.0 - alpha) * bg_crop).astype(np.uint8)
    background[y1:y2, x1:x2] = blended
    return background


def draw_tracker_box(img, x1, y1, x2, y2, track_id, color=(255, 0, 255)):
    """
    Draws corner-accented bounding box and track ID tag with pure OpenCV.
    """
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    corner_len = min(15, max(5, (x2 - x1) // 4), max(5, (y2 - y1) // 4))

    # Corner accents
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, 4)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, 4)
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, 4)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, 4)
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, 4)
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, 4)
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, 4)
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, 4)

    # Label pill
    label = f"ID: {int(track_id)}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ty = max(y1 - 6, th + 4)
    cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), color, -1)
    cv2.putText(img, label, (x1 + 3, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
