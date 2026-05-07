"""
Person D - Visualization Module
================================
Responsibility: Draw bounding boxes and labels on the frame.
Exam topics covered: Image processing, OpenCV drawing.
"""

import cv2
import random

# Persistent color per class so the same class always has same color
_COLOR_CACHE = {}


def _color_for(cls_name: str):
    if cls_name not in _COLOR_CACHE:
        random.seed(hash(cls_name) & 0xFFFFFFFF)
        _COLOR_CACHE[cls_name] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )
    return _COLOR_CACHE[cls_name]


def draw_detections(frame, detections):
    """Draw boxes + labels on a copy of the frame."""
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cls = det["class"]
        conf = det["conf"]
        color = _color_for(cls)

        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"{cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

    # Summary at top-left
    cv2.putText(
        out,
        f"Objects: {len(detections)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    return out
