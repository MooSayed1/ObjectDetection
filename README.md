# Object Detection with YOLOv8

Real-time multi-class object detection on **images, videos and webcam**, powered
by a pretrained YOLOv8 Convolutional Neural Network (80 COCO classes).

Built as a 6-person college Computer Vision project — each module is owned by
one team member.

---

## Demo

| Input | Output |
|-------|--------|
| `data/street.jpg` | 1 bus + 3 people detected (conf 0.83 – 0.87) |
| `data/zidane.jpg` | 2 people detected (conf 0.82 – 0.84) |
| Webcam | Live person detection at ~10 FPS on CPU |

Annotated results are written to `outputs/`.

---

## Features

- Pretrained **YOLOv8-nano** CNN — no training required.
- Three input modes: **image**, **video file**, **live webcam**.
- Auto-downloads sample images on first run.
- Auto-detects the working webcam index (0 – 4).
- Per-class deterministic colors for clean visualizations.
- Saves annotated images / videos to `outputs/`.

---

## Project Structure

```
ObjectDetectionProject/
├── src/
│   ├── 01_data_loader.py   # Person A: image / video / webcam input
│   ├── 02_model.py         # Person B: load pretrained YOLOv8
│   ├── 03_detector.py      # Person C: run inference, return detections
│   ├── 04_visualizer.py    # Person D: draw boxes + labels
│   ├── 05_app.py           # Person E: CLI app, glues everything
│   └── 06_metrics.py       # Person F: FPS / per-class statistics report
├── data/                   # sample images (auto-downloaded)
├── outputs/                # annotated results
├── docs/
│   ├── PROJECT_DOC.md      # full team document & talking points
│   └── CHEATSHEET.md       # what each member says to the doctor
├── requirements.txt
└── README.md
```

---

## Setup

Requires Python 3.9+.

```bash
git clone <your-repo-url>
cd ObjectDetectionProject
pip install -r requirements.txt
```

The first run will auto-download `yolov8n.pt` (~6 MB).

---

## Usage

```bash
cd src

# 1) Image mode (auto-downloads sample images on first run)
python 05_app.py --mode image
python 05_app.py --mode image --source ../data/street.jpg

# 2) Video mode  (saves outputs/video_result.mp4)
python 05_app.py --mode video --source ../data/your_video.mp4

# 3) Webcam mode  (press 'q' to quit)
python 05_app.py --mode webcam
```

Add `--no-show` to disable the GUI window (useful on headless servers).

---

## Pipeline

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  01_data     │──▶│  02_model    │──▶│  03_detector │──▶│  04_visual.  │──▶│  05_app      │
│  (Person A)  │   │  (Person B)  │   │  (Person C)  │   │  (Person D)  │   │  (Person E)  │
│ image/video/ │   │ load YOLOv8  │   │ run inference│   │ draw boxes & │   │ CLI: image / │
│ webcam input │   │ pretrained   │   │ return dets  │   │ class labels │   │ video / cam  │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

---

## Team & Responsibilities

| Member    | Module                  | Responsibility                                         |
|-----------|-------------------------|--------------------------------------------------------|
| Person A  | `01_data_loader.py`     | Read images, open video files, open webcam            |
| Person B  | `02_model.py`           | Load the pretrained YOLOv8 CNN                        |
| Person C  | `03_detector.py`        | Run inference, return clean detection dicts          |
| Person D  | `04_visualizer.py`      | Draw bounding boxes and class labels                  |
| Person E  | `05_app.py`             | CLI integration, image/video/webcam demo             |
| Person F  | `06_metrics.py`         | FPS measurement, per-class counts, performance report |

See [`docs/PROJECT_DOC.md`](docs/PROJECT_DOC.md) for the full team document and
[`docs/CHEATSHEET.md`](docs/CHEATSHEET.md) for talking points each member can use
during the oral exam.

---

## How It Works (short version)

YOLO ("You Only Look Once") is a **one-stage CNN object detector**:

1. The image is resized to 640×640 and pushed through a CNN backbone.
2. The detection head predicts, for each grid cell, candidate bounding boxes
   `(x, y, w, h)`, an objectness score and class probabilities.
3. **Non-Maximum Suppression** removes overlapping duplicate boxes.
4. The output is a list of `(box, class, confidence)` tuples.

We use the **nano** variant (`yolov8n`, ~3.2 M parameters) because it runs in
real time even on a CPU.

---

## Connection to Course Topics

| Course topic           | Where it appears in this project                                        |
|------------------------|--------------------------------------------------------------------------|
| **NN from Scratch**    | Backbone of YOLO is convolutions + activations — same neurons, deeper.  |
| **NN PyTorch**         | YOLOv8 is implemented in PyTorch (called via `model.predict`).          |
| **CNN on FMNIST**      | Same idea — CNN does feature extraction + classification.               |
| **Harris / SIFT / HOG**| Classical hand-crafted features — replaced by CNN feature maps.         |
| **Object Detection**   | The final task — locate and classify multiple objects per image.        |

---

## Requirements

```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
```

---

## License

Educational project — free to use and modify.
