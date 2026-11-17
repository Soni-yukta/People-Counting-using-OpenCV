# Comprehensive Project Audit: People Counting using OpenCV

**Date:** November 9, 2026  
**Target Repository:** `People-Counting-using-OpenCV-main`  
**Inspected Files:** `main.py`, `sort.py`, `readme.md`, `graphics-1.png`, `mask-1.png`, `people.mp4`

---

## 1. Executive Summary

A comprehensive architectural and functional inspection was conducted on the **People Counting using OpenCV** repository. The project intends to detect and track pedestrians traversing upward and downward escalators using YOLOv8 for object detection and the SORT (Simple Online and Realtime Tracking) algorithm for tracking.

**Current Operational Status:** **NON-FUNCTIONAL (Cannot Run)**.  
The project in its current state fails to execute due to:
1. Missing core runtime packages (`ultralytics`, `opencv-python`, `cvzone`, `filterpy`, `scikit-image`, `scipy`).
2. Broken dependencies and obsolete imports in `sort.py` (`from skimage import io` and `matplotlib.use('TkAgg')` at module scope).
3. Critical runtime exception on video completion (`img` is `None` when EOF is reached, crashing OpenCV).
4. Unchecked relative path assumptions exacerbated by a nested folder structure.

---

## 2. Project Structure & Asset Inventory

### 2.1 File System Organization
The repository has an anomalous nested directory structure:
```text
C:\Users\yukta\Downloads\People-Counting-using-OpenCV-main\
└── People-Counting-using-OpenCV-main\
    ├── graphics-1.png       (9,285 bytes)
    ├── main.py              (3,994 bytes)
    ├── mask-1.png           (21,939 bytes)
    ├── people.mp4           (11,596,620 bytes)
    ├── readme.md            (1,031 bytes)
    └── sort.py              (11,005 bytes)
```
- If commands are executed from the repository root without entering the inner directory, all relative paths (`"people.mp4"`, `"mask-1.png"`, `"graphics-1.png"`) fail to resolve.

### 2.2 Asset Details & Inspections

| Asset | Type | Dimensions / Specs | Details & Verification |
| :--- | :--- | :--- | :--- |
| **`people.mp4`** | Video File | 1280 x 720 px, 632 frames, ~21.12 sec, ~29.92 FPS | Camera footage of a shopping center / transit hub with two escalators: left escalator moving upward, right escalator moving downward. |
| **`mask-1.png`** | Image (Mask) | 1280 x 720 px, RGBA mode (4 channels) | Binary region-of-interest (ROI) mask: 477,107 black pixels (background) and 442,893 white pixels (escalators + stairwell). |
| **`graphics-1.png`** | Image (Overlay) | 550 x 122 px, Palette (P) mode with transparency | Dashboard banner displaying two counters ("UP" and "DOWN") with stylized card backgrounds. |

---

## 3. Existing Architecture Analysis

### 3.1 Processing Pipeline
```
[people.mp4 Frame (1280x720)]
         │
         ├──► cv2.bitwise_and(img, mask) ──► Masked Escalator Region
         │                                            │
         │                                            ▼
         │                                  YOLOv8 Detection
         │                                  (class == 'person', conf > 0.3)
         │                                            │
         │                                            ▼
         │                                  SORT Tracker Update
         │                                  (Kalman Box Filter + IoU Matching)
         │                                            │
         ▼                                            ▼
Overlay Graphics (cvzone)  ◄──────────────── Tracked Bounding Boxes & IDs
         │
         ├──► Proximity Check against limitsUp [103, 161, 296, 161]
         ├──► Proximity Check against limitsDown [527, 489, 735, 489]
         │
         ▼
cv2.imshow("Image", img)
```

### 3.2 Detection Component (`main.py`)
- Uses Ultralytics YOLO (`YOLO("yolov8n.pt")`).
- Passes `imgRegion` (the masked frame) to YOLO so inference is constrained to escalator zones.
- Filters predictions for `classNames[cls] == "person"` and `conf > 0.3`.
- Gathers detections into a `(N, 5)` NumPy matrix `[x1, y1, x2, y2, conf]`.

### 3.3 Tracking Component (`sort.py`)
- Standard academic implementation of **SORT (Simple Online and Realtime Tracking)** (Bewley et al., 2016).
- State space: constant-velocity model with 7 state variables: `[u, v, s, r, u', v', s']^T` (centroid `u, v`, scale/area `s`, aspect ratio `r`, and respective velocities).
- Association: Bounding box IoU cost matrix solved by `lap.lapjv` or `scipy.optimize.linear_sum_assignment`.

---

## 4. Verification & Diagnostic Test Results

Tests were performed against the active system environment without mock data:

### Test 1: Python Environment Check
- System default Python: `Python 3.14.3` (`C:\Users\yukta\AppData\Local\Programs\Python\Python314\python.exe`).
- Secondary Python: `Python 3.11` (`C:\Users\yukta\AppData\Local\Programs\Python\Python311\python.exe`).
- Active Virtual Environments: **None detected**.
- Key finding: Python 3.14 has limited pre-built binary wheel support for packages such as `torch`, `opencv-python`, and C++ extensions like `lap`.

### Test 2: Dependency Verification (`pip list`)
- `ultralytics`: **MISSING**
- `opencv-python` / `cv2`: **MISSING**
- `cvzone`: **MISSING**
- `filterpy`: **MISSING**
- `scikit-image` (`skimage`): **MISSING**
- `scipy`: **MISSING**
- `torch` / `torchvision`: **MISSING**

### Test 3: Execution Test of `main.py`
Command: `python main.py`  
Output:
```text
Traceback (most recent call last):
  File "...\main.py", line 2, in <module>
    from ultralytics import YOLO
ModuleNotFoundError: No module named 'ultralytics'
```

### Test 4: Execution Test of `sort.py`
Command: `python -c "import sort"`  
Output:
```text
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import sort
  File "...\sort.py", line 9, in <module>
    from skimage import io
ModuleNotFoundError: No module named 'skimage'
```

---

## 5. Identified Bugs, Deficiencies & Risks

### 5.1 Critical Bugs

1. **Crash on Video Termination (Missing EOF Handling)**
   - Line 35: `success, img = cap.read()`
   - Line 36: `imgRegion = cv2.bitwise_and(img, mask)`
   - When the 632-frame video finishes, `success` becomes `False` and `img` becomes `None`.
   - Without an `if not success: break` check, `cv2.bitwise_and` raises an unhandled OpenCV assertion failure: `(-215:Assertion failed) (src1.size == src2.size && src1.type() == src2.type())`, abruptly crashing the program.

2. **Per-Frame Disk I/O Bottleneck**
   - Line 38: `imgGraphics = cv2.imread("graphics-1.png", cv2.IMREAD_UNCHANGED)` is executed inside the `while True` loop every single frame.
   - Reading the image file from disk 30 times per second incurs unnecessary file system latency and CPU overhead.

3. **Direction-Agnostic Counting Logic (False Positives)**
   - Line 77: `if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:`
   - Line 82: `if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:`
   - Counting is determined solely by centroid presence within a static vertical corridor (`±15 px`).
   - There is no trajectory vector check (e.g., verifying whether $cy_t - cy_{t-1} < 0$ for UP or $> 0$ for DOWN). A stationary person, someone walking the wrong direction, or someone lingering in the band can trigger false counts.

4. **Frame-Skip / Speed Vulnerability (Missed Counts)**
   - Fast-moving pedestrians or frames where YOLO briefly drops detection can cause centroid displacement exceeding 30 pixels between tracked frames.
   - If a person hops from $cy = 140$ to $cy = 180$, they bypass the $161 \pm 15$ window completely without triggering a count.
   - Robust line crossing requires geometric line segment intersection between $(cx_{t-1}, cy_{t-1}) \to (cx_t, cy_t)$ and the boundary segment.

5. **Visual Feedback Glitch**
   - Lines 62–63 draw the red line on every frame.
   - Lines 80 and 85 draw a green line when a person is in the band.
   - Because the line color is reset to red at the beginning of the next frame, the green flash lasts for exactly one frame (~33ms), appearing as an unnoticeable flicker rather than sustained feedback.

6. **Hot-Loop Console Logging Overhead**
   - Line 68: `print(result)` outputs tracking vectors to stdout on every frame for every detected person.
   - Windows terminal I/O for stdout is notoriously slow and causes severe frame drops.

### 5.2 Dependency & Architecture Deficiencies

1. **Unnecessary and Problematic Dependencies in `sort.py`**
   - Top-level `from skimage import io` and `matplotlib.use('TkAgg')` are imported even when SORT is only used as a tracking class in `main.py`.
   - `filterpy.kalman.KalmanFilter` adds an external dependency for a standard linear Kalman filter that can easily be handled using native NumPy or `cv2.KalmanFilter`.
   - `lap` requires C++ compilers on Windows and often fails to build.
2. **Hardcoded Magic Numbers & Geometry**
   - Coordinates `limitsUp = [103, 161, 296, 161]` and `limitsDown = [527, 489, 735, 489]` are hardcoded for `people.mp4` (1280x720).
   - Graphic placement `(730, 260)` and text coordinates `(929, 345)`, `(1191, 345)` are fixed. Any video with different dimensions will cause out-of-bounds errors or distorted positioning.
3. **Missing Model Weights**
   - `yolov8n.pt` is not stored in the repository. Running `main.py` requires an automated download from GitHub, which fails if network access is restricted or behind an unconfigured proxy.
4. **Documentation Discrepancies (`readme.md`)**
   - The README states:
     - Prerequisites: `pip install numpy opencv-python imutils` (omits `ultralytics`, `cvzone`, `filterpy`, `scipy`).
     - Mentions "MobileNet SSD", which is not present in the codebase.
     - Claims real-time webcam and static image capabilities, yet `main.py` hardcodes `people.mp4` with no CLI flags.

---

## 6. Proposed Implementation Plan

To transform this repository into a robust, professional, and maintainable computer vision solution, the following phases are proposed:

### Phase 1: Environment & Dependency Stabilization
- Standardize on **Python 3.11** (which has verified, mature binary wheels for PyTorch, TorchVision, and OpenCV on Windows).
- Create a `requirements.txt` specifying exact, compatible dependencies:
  - `opencv-python`
  - `ultralytics`
  - `numpy`
  - `scipy`
  - `pillow`
- Provide clean fallbacks for `cvzone` (pure OpenCV implementations for overlaying alpha-transparent PNGs and bounding boxes) to eliminate unneeded dependencies.

### Phase 2: Modernized, Self-Contained Tracker
- Refactor or replace `sort.py`:
  - Eliminate top-level imports of `skimage` and `matplotlib`.
  - Provide a standalone, lightweight Kalman Filter tracker implementation using pure `numpy` and `scipy.optimize.linear_sum_assignment`.
  - Prevent dependency on `filterpy` and `lap`.
  - Reset tracker IDs cleanly between video runs.

### Phase 3: Mathematical Trajectory & Line Crossing Algorithm
- Replace the fragile bounding box $\pm 15$ px proximity check with a **2D Line Segment Intersection Algorithm**:
  - Track each person's trajectory history: $P_{t-1} = (cx_{t-1}, cy_{t-1})$ and $P_t = (cx_t, cy_t)$.
  - Test if segment $P_{t-1}P_t$ intersects line segment $L_1L_2$.
  - Compute the cross-product vector sign to determine the **true direction of travel** (Upward vs. Downward).
  - Add debounce / state locking so an ID cannot trigger multiple counts.
  - Maintain persistent green highlight state for counting lines (e.g. 10 frames) so visual feedback is clear and professional.

### Phase 4: Structural & Performance Optimization
- Fix nested directory structure so the project can be run directly from the workspace root.
- Move disk operations outside the main loop (`imgGraphics = cv2.imread(...)` loaded once).
- Add proper video stream termination handling (`if not success: break`).
- Remove frame-by-frame stdout `print()` statements; replace with an optional `--verbose` flag or a clean FPS counter overlay.
- Add command-line argument support (`argparse`) allowing:
  - Custom video file, webcam index (`0`), or RTSP stream (`--source`).
  - Model selection (`--weights yolov8n.pt`).
  - Confidence threshold (`--conf 0.3`).
  - Headless mode / video recording output (`--save-out output.mp4`).

### Phase 5: Comprehensive Documentation & Testing
- Update `README.md` with accurate setup commands, usage instructions, argument documentation, and architecture diagrams.
- Add an automated verification script to validate video loading, detection, tracking, and counting without GUI dependency.io























































































































































































































































 