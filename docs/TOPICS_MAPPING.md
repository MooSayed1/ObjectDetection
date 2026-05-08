# Course Topics — What We Actually Used

This document maps the 10 exam topics to our Object Detection project.
Be honest in the exam: we only **directly use** a few of these topics in code.
The rest are **conceptually related** — say so when asked.

Legend:
- ✅ **Directly used in code** — runs every time we execute the project.
- 🧠 **Conceptually used** — the underlying idea is what YOLO does internally, but we didn't write it ourselves.
- ❌ **Not used** — be ready to explain why it's not the right fit for this project.

---

## Topic Cheat-Sheet (one paragraph each)

### 1) NN from Scratch
A **Neural Network** is a stack of layers. Each *neuron* computes
`output = activation(W·x + b)` — a weighted sum of its inputs followed by a
non-linear function (sigmoid / ReLU). "From scratch" means we wrote the
forward pass, the loss, and the **backpropagation** (computing gradients
with the chain rule and updating weights with gradient descent) using only
NumPy — no framework. The point is to *see* what a framework hides.

### 2) NN PyTorch
The same idea but using **PyTorch**, a deep-learning framework. PyTorch
gives us `nn.Linear`, `nn.ReLU`, automatic differentiation (`loss.backward()`)
and an optimizer (`Adam`). We define the model in `forward()` and PyTorch
handles all the gradient math on GPU. Much shorter, faster, and used in
every modern deep-learning project.

### 3) CNN on FMNIST
A **Convolutional Neural Network** classifies the **Fashion-MNIST** dataset
(28×28 grayscale clothing images, 10 classes). A CNN replaces the dense
layers with **convolutions** — small filters that slide over the image and
detect local patterns (edges, textures). The standard recipe is
`Conv → BatchNorm → Activation → Pool`, repeated, then a fully-connected
layer outputs the class probabilities. CNNs work much better on images than
plain NNs because they *share weights* and respect spatial structure.

### 4) Harris Corner Detector
A classical algorithm to find **corners** in an image (places where the
gradient changes in two directions, like the corner of a window or a
chessboard square). It computes a `2×2` matrix of gradient products per
pixel and a "cornerness" score; high scores = corner. Useful for tracking
and matching, but it only finds *where* an interesting point is — not what
it represents.

### 5) SIFT (Scale-Invariant Feature Transform)
A classical method that does two things: (a) finds **keypoints** that are
invariant to scale and rotation (built using a Difference-of-Gaussians
pyramid), and (b) computes a 128-dimensional **descriptor** for each
keypoint based on local gradient orientations. SIFT descriptors of the same
real-world point look similar even if the image is rotated, scaled or
slightly distorted — perfect for matching the same object in two photos
(used in panorama stitching and image retrieval).

### 6) HOG (Histogram of Oriented Gradients) on Image
A classical *image-level* feature. Split the image into small cells, compute
the gradient direction & magnitude in each, and build a **histogram of
orientations**. Concatenate all histograms → a long feature vector that
describes the whole image's shape structure. Famous combo: **HOG + linear
SVM** for pedestrian detection (Dalal & Triggs, 2005). It was *the*
state-of-the-art object detector before deep learning.

### 7) RANSAC (Random Sample Consensus)
A robust algorithm to **fit a model to data with outliers**. Repeatedly:
pick a random minimal subset → fit a model (line, homography, etc.) →
count how many points agree (inliers). Keep the model with the most
inliers. Used heavily in computer vision when you have noisy correspondences,
e.g., to find the transformation between two images for panorama stitching.

### 8) Panorama
The task of **stitching multiple overlapping photos** into one wide image.
Pipeline: detect keypoints with SIFT in both images → match descriptors →
use **RANSAC** to find a robust **homography** (a 3×3 transform) → warp one
image onto the other → blend. Combines SIFT + RANSAC + image warping.

### 9) BoVW (Bag of Visual Words)
The classical pipeline for **image classification** before CNNs.
(1) Extract SIFT descriptors from all training images.
(2) Cluster all descriptors with **K-Means** to build a "vocabulary" of
visual words.
(3) Represent each image as a **histogram** counting how many descriptors
fell into each cluster.
(4) Train a classifier (SVM) on these histograms. The same idea as
"bag of words" in NLP, but with image patches instead of text words.

### 10) Clustering
**Unsupervised learning** that groups similar items together with no
labels. Most common algorithm: **K-Means** — pick K random centers,
assign each point to the nearest center, move centers to the mean of
their assigned points, repeat until stable. Used inside BoVW (to build the
vocabulary), in color quantization (reducing colors in an image), and in
older YOLO versions to choose default anchor-box sizes from the training
set.

---

## Quick Summary Table

| # | Topic            | Status | Where / Why                                                          |
|---|------------------|--------|----------------------------------------------------------------------|
| 1 | NN from Scratch  | 🧠     | Same neuron math is inside YOLO, just not coded by us.              |
| 2 | NN PyTorch       | ✅     | YOLOv8 is a PyTorch model. `model.predict()` calls a PyTorch CNN.    |
| 3 | CNN on FMNIST    | ✅     | YOLOv8's backbone is exactly Conv → BN → Activation → Pool, deeper.  |
| 4 | Harris           | ❌     | Corner detector — replaced by CNN feature maps inside YOLO.          |
| 5 | SIFT             | ❌     | Hand-crafted keypoints — replaced by learned features.               |
| 6 | HOG on Image     | ❌     | Pre-deep-learning detector pipeline (HOG + SVM).                     |
| 7 | RANSAC           | ❌     | Geometry/model fitting — different problem (panorama, not detection).|
| 8 | Panorama         | ❌     | Image stitching task — not what we're doing.                         |
| 9 | BoVW             | ❌     | Old image-classification pipeline — replaced by deep features.       |
|10 | Clustering       | ❌     | Used in BoVW and color quantization — not in our pipeline.           |

> **One-line summary:** We use the *modern* equivalent (CNN / PyTorch) of the
> classical topics (HOG / SIFT / BoVW). The classical methods are the
> "before" picture, our project is the "after".

---

## ✅ Topics we DIRECTLY use

### 2) NN PyTorch

**Where:** `02_model.py` — `YOLO("yolov8n.pt")`.

YOLOv8 is built on PyTorch. When we call `model.predict(frame)`, internally
PyTorch performs a **forward pass** on a deep CNN and returns tensors with
the bounding boxes and class probabilities.

**What to say:**
> "Our model is a PyTorch CNN. `model.predict()` runs a forward pass through
> ~100 PyTorch layers (Conv2d, BatchNorm2d, SiLU activations) — the same
> building blocks we used in `NNPytorch`."

---

### 3) CNN on FMNIST

**Where:** the YOLOv8 backbone (CSPDarknet) is structurally the same as the
small CNN we built for Fashion-MNIST, just much deeper and wider.

| FMNIST CNN we wrote                  | YOLOv8 backbone                              |
|--------------------------------------|----------------------------------------------|
| 2–3 Conv layers                      | ~50 Conv layers across multiple scales       |
| MaxPool / Stride 2                   | Strided Conv (same down-sampling idea)       |
| ReLU                                 | SiLU (Swish, smoother ReLU variant)          |
| Output: 10 class probabilities       | Output: 80 class probabilities + 4 box coords|

**What to say:**
> "The architecture is the same recipe as `CNNonFMNIST` — Conv → BN → activation
> → down-sample, repeat. We're just classifying *and localizing* on natural
> images instead of 28×28 grayscale."

---

## 🧠 Topics we use CONCEPTUALLY (inside the model)

### 1) NN from Scratch

**Where:** every neuron inside YOLO does `output = activation(W·x + b)` —
exactly what we coded by hand. We just didn't write it ourselves; PyTorch /
Ultralytics do it on GPU.

**What to say:**
> "We didn't reimplement the neurons, but every layer in YOLO is the same
> `W·x + b → activation` we wrote in `NNfromScratch`. PyTorch handles the
> backprop and weight updates that we coded manually in the lab."

---

## ❌ Topics we do NOT use (and why)

### 4) Harris Corner Detector

Detects **corners** in an image (intersections of edges). It's a feature
*detector*, not a *classifier*. Inside YOLO, the early Conv layers learn
edge / corner detectors automatically — Harris is replaced by *learned*
filters.

**If asked "could we have used Harris?":**
> "Harris finds keypoints but doesn't tell you what's in the image — we need
> classes (person, car, bus). Harris alone can't do object detection."

---

### 5) SIFT (Scale-Invariant Feature Transform)

SIFT finds keypoints + describes them with a 128-D descriptor — used for
matching the **same object** across two images (e.g., panorama, image
retrieval). It's not designed for "what category is this object?".

**If asked "could SIFT detect objects?":**
> "SIFT is great for matching, not classification. You'd need a database of
> SIFT descriptors per class (BoVW), and even then accuracy is much worse
> than a CNN."

---

### 6) HOG (Histogram of Oriented Gradients)

HOG was the **standard object detector before deep learning** (HOG + SVM,
Dalal & Triggs 2005, used famously for pedestrian detection). It computes
gradient orientation histograms in image cells, then a linear SVM classifies
"person / not person" with a sliding window.

**Why not used here:**
- HOG works for one class at a time (one SVM per class).
- We need 80 classes — would need 80 sliding windows. Slow & inaccurate.
- YOLO replaces both the feature (HOG) and the classifier (SVM) with one CNN.

**If asked to compare HOG vs YOLO:**
> "HOG is a hand-crafted feature with a separate SVM per class. YOLO learns
> features end-to-end and predicts all 80 classes in one pass. Same problem,
> 1000× faster, much more accurate."

---

### 7) RANSAC (Random Sample Consensus)

RANSAC fits a **geometric model** (line, homography, fundamental matrix)
robustly when data has outliers. Used in panorama stitching, structure-from-
motion, calibration.

**Why not used:** Object detection has no geometric model to fit — it's a
pattern-recognition problem.

---

### 8) Panorama

Image stitching: take multiple overlapping photos and combine them into one
wide image. Pipeline = SIFT keypoints → match → RANSAC homography → warp.

**Why not used:** Different task. We process one image at a time, not a
sequence to stitch.

---

### 9) BoVW (Bag of Visual Words)

The pre-CNN approach to **image classification**:
1. Extract SIFT keypoints from many images.
2. Cluster the descriptors with K-Means → "visual words" (a dictionary).
3. Represent each image as a histogram of visual words.
4. Train an SVM on histograms.

**Why not used:** YOLO replaces all four steps. The CNN learns its own
features (no SIFT), its own representation (no clustering), and its own
classifier (no SVM).

**If asked "how does BoVW relate?":**
> "BoVW is the classical pipeline for the same problem CNNs solve today.
> Each Conv layer in YOLO replaces SIFT + clustering with a learned filter
> bank — better and end-to-end trainable."

---

### 10) Clustering (K-Means)

Used inside BoVW, color quantization, anchor-box generation, etc.

**Tiny YOLO connection:** older YOLO versions ran K-Means on training-set
boxes to find good **anchor box sizes**. YOLOv8 dropped this — it's
**anchor-free** — so we don't run K-Means anywhere in our project.

---

## How to answer "What course topics did you use?" in 30 seconds

> "Directly: **NN PyTorch** and **CNN on FMNIST** — our detector is a deep
> PyTorch CNN with the same Conv-BN-activation blocks we built for FMNIST.
>
> Conceptually: **NN from Scratch** — every neuron inside is the same
> `W·x + b → activation` we wrote by hand.
>
> The classical topics (**HOG, SIFT, Harris, BoVW, Clustering**) are the
> *old way* to do detection / classification. YOLO replaces them with
> *learned* features. We can explain how each one would solve a piece of
> the problem, but the CNN does it all in one shot.
>
> **RANSAC and Panorama** are for image stitching, which is a different
> task — not used here."
