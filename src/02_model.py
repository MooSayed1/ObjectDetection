"""
Person B - Model Module
=======================
Responsibility: Load a pretrained YOLOv8 object detector.
Exam topics covered: CNN architectures (CNNonFMNIST), NNPytorch.

YOLOv8 = Convolutional Neural Network trained on COCO dataset (80 classes).
We do NOT train from scratch (1 day deadline) - we use the pretrained weights.

Why YOLO?
    - It's a single-stage CNN object detector.
    - Inputs an image, outputs bounding boxes + class labels + confidence.
    - State-of-the-art accuracy and very fast.
"""

from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"  # 'n' = nano (smallest, fastest). Other: s, m, l, x.


def load_model(weights: str = MODEL_NAME):
    """Load pretrained YOLOv8 model."""
    print(f"[Model] Loading {weights} ...")
    model = YOLO(weights)
    print(
        f"[Model] Loaded. Classes: {len(model.names)} (e.g. {list(model.names.values())[:5]} ...)"
    )
    return model


if __name__ == "__main__":
    m = load_model()
    print("[Model] Ready.")
