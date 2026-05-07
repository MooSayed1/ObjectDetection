# Object Detection Project — Team Documentation

> **Goal:** Build an Object Detection system that works on **images, videos and webcam**, detecting multiple object classes (person, car, bus, dog, etc.).
>
> **Team size:** 5 members. Each member owns one module of the pipeline.
>
> **Approach:** Use the pretrained **YOLOv8** model (a Convolutional Neural Network already trained on the **COCO** dataset, 80 object categories). We do *inference only* — no training required.

---

## 1. Why YOLOv8?

| Property            | Value                                                |
|---------------------|------------------------------------------------------|
| Architecture        | One-stage CNN object detector                        |
| Backbone            | CSPDarknet (Convolution + Batch-Norm + SiLU)         |
| Trained on          | COCO dataset (~118k images, 80 classes)              |
| Output              | Bounding boxes + class labels + confidence scores    |
| Variant used        | `yolov8n` (nano, ~3.2M parameters, very fast on CPU) |

YOLO ("You Only Look Once") processes the **whole image in a single forward pass** of a CNN and predicts boxes directly — that's why it is fast enough for real-time webcam detection.

### How YOLO works (high level)

1. **Input:** image of size 640×640 (auto-resized).
2. **CNN backbone** extracts feature maps at multiple scales.
3. **Detection head** predicts, for each grid cell, candidate bounding boxes `(x, y, w, h)`, an *objectness* score, and class probabilities.
4. **Non-Maximum Suppression (NMS)** removes overlapping duplicate boxes.
5. Final output: a list of `(box, class, confidence)`.

---

## 2. Pipeline

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  01_data     │──▶│  02_model    │──▶│  03_detector │──▶│  04_visual.  │──▶│  05_app      │
│  (Person A)  │   │  (Person B)  │   │  (Person C)  │   │  (Person D)  │   │  (Person E)  │
│ image/video/ │   │ load YOLOv8  │   │ run inference│   │ draw boxes & │   │ CLI: image / │
│ webcam input │   │ pretrained   │   │ return dets  │   │ class labels │   │ video / cam  │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

---

## 3. Task Division (each member owns one file)

### Person A — Data Module · `src/01_data_loader.py`

**Job:** provide images, video files and webcam streams to the rest of the pipeline.

Functions implemented:
- `download_sample_images()` — fetches 2 sample images from the internet.
- `load_image(path)` — reads an image with OpenCV.
- `get_video_capture(source)` — opens a video file or webcam (`source=0`).

**Exam topics to mention to the doctor**
- Image as a 3-D numpy array `(H, W, 3)` in BGR.
- Resizing / normalization (done internally by YOLO).
- Difference between image, video and live capture.

---

### Person B — Model Module · `src/02_model.py`

**Job:** load the pretrained YOLOv8 detector.

Functions implemented:
- `load_model(weights="yolov8n.pt")` — downloads (first run) and loads the model.

**Exam topics to mention**
- CNN architecture: convolution → batch-norm → activation → pooling.
- Why pretrained weights? (Transfer learning: avoid training from scratch.)
- 80 COCO classes — model knows them all out of the box.
- `yolov8n / s / m / l / x` trade-off (size vs. accuracy).

---

### Person C — Detector Module · `src/03_detector.py`

**Job:** run the model on a frame and return clean detection dictionaries.

Function implemented:
- `detect(model, frame, conf_threshold=0.4)` — returns
  `[{"box": (x1, y1, x2, y2), "class": "person", "conf": 0.87}, ...]`.

**Exam topics to mention**
- Confidence score (sigmoid / probability) and the threshold.
- Non-Maximum Suppression (NMS): removes overlapping boxes for the same object.
- Why we pass the raw frame and let YOLO handle resizing internally.

---

### Person D — Visualization Module · `src/04_visualizer.py`

**Job:** draw the detections on the frame.

Function implemented:
- `draw_detections(frame, detections)` — returns a copy of the frame with
  colored bounding boxes, class names, confidences and an *Objects: N* counter.

**Exam topics to mention**
- OpenCV drawing primitives (`rectangle`, `putText`).
- Per-class deterministic colors using a hash → consistent across frames.
- Don't draw on the original frame — always copy.

---

### Person E — Application Module · `src/05_app.py`

**Job:** Tie everything together. Provide a CLI with three modes:

| Mode    | Command                                             |
|---------|-----------------------------------------------------|
| Image   | `python 05_app.py --mode image --source img.jpg`    |
| Video   | `python 05_app.py --mode video --source vid.mp4`    |
| Webcam  | `python 05_app.py --mode webcam`                    |

For video mode the annotated output is saved to `outputs/video_result.mp4`.

**Exam topics to mention**
- End-to-end pipeline integration.
- `cv2.VideoCapture` / `cv2.VideoWriter`.
- FPS measurement (frames / elapsed time).

---

## 4. Sample Result (already generated)

Run on `data/street.jpg`:

```
[Model] Loaded. Classes: 80
[App] Detected 4 objects. Saved -> outputs/image_result.jpg
   bus    0.87
   person 0.87
   person 0.85
   person 0.83
```

The saved image shows colored bounding boxes around the bus and three people.

---

## 5. Connection to Course Topics (for the oral exam)

| Course topic         | How it appears in this project                                   |
|----------------------|------------------------------------------------------------------|
| **NN from Scratch**  | Backbone of YOLO is convolution+activation — same neurons we wrote by hand, just stacked deeper. |
| **NN PyTorch**       | YOLOv8 is implemented in PyTorch (the `model.predict` we call).  |
| **CNN on FMNIST**    | Same idea — CNN does feature extraction + classification, but here on natural images instead of FMNIST. |
| **Harris / SIFT / HOG** | Classical hand-crafted feature detectors — replaced today by CNN feature maps that YOLO learns automatically. |
| **RANSAC / Panorama** | Geometry / robust fitting — *not* used here, but useful contrast: YOLO is data-driven, RANSAC is model-driven. |
| **BoVW / Clustering** | Old "bag of visual words" pipeline for image classification — replaced by deep features inside YOLO. |
| **Object detection** | The final task — locate **and** classify multiple objects per image. |

> **One-line summary you can say to the doctor:**
> *"We built a real-time object detector by combining a pretrained YOLOv8 CNN with an OpenCV-based input/visualization pipeline. The CNN replaces classical features like HOG/SIFT and learns its own features end-to-end."*

---

## 6. How to Run (quick reference)

```bash
pip install -r requirements.txt
cd src
python 05_app.py --mode image                       # uses a sample image
python 05_app.py --mode image  --source ../data/street.jpg
python 05_app.py --mode video  --source ../data/sample.mp4
python 05_app.py --mode webcam
```

Outputs are written to `outputs/`.
