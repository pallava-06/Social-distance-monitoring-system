# 🚧 Social Distancing Detector using YOLOv11

A real-time computer vision project that detects people in video footage and flags **social distancing violations** based on the distance between them, using **Ultralytics YOLOv11** for person detection and OpenCV for visualization.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLO](https://img.shields.io/badge/YOLOv11-Ultralytics-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

This project processes a video file frame-by-frame, detects all people present using a YOLOv11 object detection model, and calculates the **Euclidean distance** between the centroids of every pair of detected individuals. If two people are closer than a configurable pixel threshold, they are flagged as **violating social distancing** and highlighted in red. The annotated video is saved to disk and can optionally be previewed live.

---

## 📸 Output Screenshots

> Replace the placeholders below with your own screenshots once you've run the script. Save your images inside a `screenshots/` folder in the repo root, then reference them as shown.

```
.
├── screenshots/
│   ├── safe_distance.png
│   └── violation_detected.png
```

| Safe Distancing | Violation Detected |
|---|---|
| ![Safe distancing example](screenshots/safe_distance.png) | ![Violation detected example](screenshots/violation_detected.png) |

You can also embed a short GIF or clip of the output video for a quick preview:

```markdown
![Demo](screenshots/demo.gif)
```

---

## ✨ Features

- 🧍 **Person Detection** — Uses YOLOv11 (nano variant by default) for fast, accurate detection.
- 📏 **Distance Estimation** — Computes pairwise centroid distances using `scipy.spatial.distance`.
- 🟢🔴 **Visual Feedback** — Green boxes for safe distancing, red boxes for violations.
- 🔢 **Live Violation Counter** — Displays the number of violations on each frame.
- 🎥 **Video Export** — Saves the fully annotated output as an `.mp4` file.
- ⚙️ **Configurable Thresholds** — Easily tune confidence and distance parameters.

---

## 🧠 How It Works

1. **Load Model** — The YOLOv11 nano model (`yolo11n.pt`) is loaded via the `ultralytics` library.
2. **Read Frame** — Each frame is read from the input video and resized for consistent processing.
3. **Detect People** — The frame is passed through YOLO; detections are filtered to keep only the `person` class above a minimum confidence (`MIN_CONF`).
4. **Compute Centroids** — For every detected person, the bounding box center `(cX, cY)` is calculated.
5. **Measure Distances** — Pairwise Euclidean distances between all centroids are computed using `cdist`.
6. **Flag Violations** — Any pair of people closer than `MIN_DISTANCE` pixels is marked as violating.
7. **Annotate Frame** — Bounding boxes and centroid markers are drawn — red for violators, green for everyone else — along with a running violation count.
8. **Write & Preview** — The annotated frame is written to the output video and optionally displayed in a live window (press `q` to quit early).

---

## 🛠️ Tech Stack

| Component | Purpose |
|---|---|
| [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) | Person detection |
| OpenCV (`cv2`) | Video I/O, drawing, and display |
| NumPy | Array operations |
| SciPy (`scipy.spatial.distance`) | Centroid distance calculations |

---

## 📂 Project Structure

```
.
├── social_distancing_detector.py   # Main script
├── video.mp4                       # Input video (provide your own)
├── output.mp4                      # Generated output (created on run)
├── yolo11n.pt                      # YOLOv11 weights (auto-downloaded on first run)
├── screenshots/                    # Output screenshots/GIFs for the README
│   ├── safe_distance.png
│   └── violation_detected.png
└── README.md
```

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install ultralytics opencv-python numpy scipy
   ```

   Or, if you have a `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your input video**
   Place a video file named `video.mp4` in the project root (or update `INPUT_VIDEO` in the script).

---

## ▶️ Usage

Run the script directly:

```bash
python social_distancing_detector.py
```

- A live preview window will open showing real-time detections and violations.
- Press **`q`** at any time to stop processing early.
- The fully annotated video is saved as `output.mp4` in the project root.

> ⚠️ Note: The first run will automatically download the `yolo11n.pt` weights file via Ultralytics.

---

## 🔧 Configuration

These parameters can be adjusted at the top of the script:

| Variable | Default | Description |
|---|---|---|
| `INPUT_VIDEO` | `"video.mp4"` | Path to the source video file |
| `OUTPUT_VIDEO` | `"output.mp4"` | Path where the annotated video is saved |
| `MIN_CONF` | `0.15` | Minimum confidence score to accept a person detection |
| `MIN_DISTANCE` | `50` | Pixel distance below which two people are flagged as violating |

You can also switch to a larger, more accurate (but slower) model by changing:

```python
model = YOLO("yolo11n.pt")   # try yolo11s.pt, yolo11m.pt, yolo11l.pt, or yolo11x.pt
```

---

## ⚠️ Limitations

- **Pixel-based distance, not real-world distance** — `MIN_DISTANCE` is measured in pixels, so the appropriate threshold depends heavily on camera angle, resolution, and distance from the subjects. There is no perspective/homography correction.
- **No persistent tracking** — Each frame is processed independently, so the same person may not retain a consistent identity across frames.
- **Single-class detection** — Only the `person` class is considered; occlusion or overlapping detections can affect accuracy.
- **2D centroid approximation** — Distance is computed between bounding box centers, which can be inaccurate for people at different depths in the scene.

---

## 🚀 Future Improvements

- [ ] Add a **bird's-eye view (perspective transform)** for accurate real-world distance estimation.
- [ ] Integrate **object tracking** (e.g., DeepSORT or ByteTrack) for consistent person IDs across frames.
- [ ] Add **CLI arguments** for video path, model size, and thresholds instead of hardcoded constants.
- [ ] Support **live webcam/RTSP stream** input in addition to video files.
- [ ] Export violation statistics (e.g., CSV/JSON log of violations per frame).

---

## 📜 License

This project is licensed under the [MIT License](LICENSE) — feel free to use, modify, and distribute it.

---

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv11 model and library.
- Inspired by classic OpenCV-based social distancing detector tutorials.
