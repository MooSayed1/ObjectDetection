# Cheat-Sheet — what each member says when the doctor asks "what did YOU do?"

---

## Person A (Data) — `01_data_loader.py`

> "I handle the input side of the pipeline. My module can load a single image
> from disk with OpenCV, open a video file frame-by-frame, or open the webcam
> as a live stream. I also auto-download a couple of sample images so the demo
> runs on a fresh machine without manual setup. Images are stored as numpy
> arrays in BGR order — that's the OpenCV convention."

Likely Qs:
- *Why BGR not RGB?* OpenCV historical convention.
- *How do you read a video?* `cv2.VideoCapture` and loop `cap.read()`.

---

## Person B (Model) — `02_model.py`

> "I load the pretrained YOLOv8-nano model from Ultralytics. It's a CNN trained
> on COCO with 80 classes. We chose the *nano* variant because we have one day
> and it runs in real time even on CPU. I'm using transfer learning — the
> weights are already trained, so we don't need to re-train."

Likely Qs:
- *What is YOLO?* "You Only Look Once" — single-stage CNN detector that predicts boxes and classes in one forward pass.
- *Why pretrained?* Saves training time and gigabytes of data.
- *What's COCO?* A standard dataset of ~118k labelled images with 80 categories.

---

## Person C (Detector) — `03_detector.py`

> "My function takes the model and a frame and returns a clean list of
> detections — bounding box, class name, confidence. I apply a confidence
> threshold (0.4) to drop weak predictions. YOLO internally already does
> Non-Maximum Suppression to merge duplicate boxes."

Likely Qs:
- *What is a bounding box?* `(x1, y1, x2, y2)` — top-left and bottom-right pixel coordinates.
- *What is confidence?* Probability the box contains an object of that class.
- *What is NMS?* Non-Maximum Suppression — keep the highest-score box and remove others that overlap it (IoU > 0.5).

---

## Person D (Visualizer) — `04_visualizer.py`

> "I draw the results on a copy of the frame. Each class gets a deterministic
> color (hashed from the class name) so 'person' is always the same color
> across frames — looks consistent in videos. I also overlay the total object
> count and the confidence on each label."

Likely Qs:
- *Why copy the frame?* Don't mutate the original — pipeline cleanliness.
- *Why hash colors?* Deterministic, no flicker between frames.

---

## Person E (App) — `05_app.py`

> "I'm the integrator. I expose a simple CLI with three modes — image, video
> and webcam. For video mode I also save the annotated output as `.mp4` using
> `cv2.VideoWriter`. This is what we actually run in front of the doctor."

Likely Qs:
- *How do you stop the live demo?* Press 'q'.
- *Why a CLI?* So we can demo any of the three modes without changing code.

---

## Person F (Metrics) — `06_metrics.py`

> "I built a `MetricsTracker` class that wraps every detection call. It
> measures the inference time per frame, the total FPS, the average
> confidence across all detections, and counts how many of each class were
> seen. At the end it prints a clean report and also saves it as a `.txt`
> file in `outputs/`."

Likely Qs:
- *What's the FPS?* Depends on hardware — ~30 FPS on GPU, ~10 FPS on CPU for the nano model.
- *Why measure FPS?* To prove the system is real-time, and to compare model variants (n / s / m / l / x).
- *Difference between speed and accuracy?* FPS = how fast; confidence / mAP = how correct. They usually trade off.

---

## Common questions for the whole team

**Q: How is this different from classical methods like HOG + SVM?**
A: HOG manually designs features (gradient orientation histograms), then SVM
classifies. YOLO *learns* features automatically inside the CNN — usually
much more accurate but needs lots of training data and GPU power.

**Q: Could you train it on your own classes?**
A: Yes — `model.train(data="my.yaml", epochs=50)` — but we didn't because
of the 1-day deadline; the pretrained 80 COCO classes already cover our demo.

**Q: Where is the CNN in your code?**
A: It's hidden inside `model.predict(...)`. The model object holds the full
PyTorch CNN — same building blocks as our `CNNonFMNIST` notebook (Conv → BN
→ activation → pooling), just much deeper.
