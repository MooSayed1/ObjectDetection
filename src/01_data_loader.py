"""
Person A - Data Module
=======================
Responsibility: Provide test images, video and webcam streams to the pipeline.
Exam topics covered: Image preprocessing (NNPytorch / CNNonFMNIST).

Functions:
    - download_sample_images(): grab a few public sample images for testing.
    - load_image(path): read an image with OpenCV (BGR -> RGB).
    - get_video_capture(source): return a cv2.VideoCapture for video file or webcam.
"""

import os
import cv2
import urllib.request

SAMPLE_IMAGES = {
    "street.jpg": "https://ultralytics.com/images/bus.jpg",
    "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def download_sample_images():
    """Download a couple of sample images if they don't already exist."""
    paths = []
    for name, url in SAMPLE_IMAGES.items():
        out = os.path.join(DATA_DIR, name)
        if not os.path.exists(out):
            print(f"[Data] Downloading {name} ...")
            try:
                urllib.request.urlretrieve(url, out)
            except Exception as e:
                print(f"[Data] Failed to download {name}: {e}")
                continue
        paths.append(out)
    return paths


def load_image(path):
    """Read image from disk and return as numpy array (BGR)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def get_video_capture(source=0):
    """
    source = int  -> webcam index (auto-tries 0..4 if requested index fails)
    source = str  -> path to a video file
    """
    # Video file path
    if isinstance(source, str):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        return cap

    # Webcam: try the requested index first, then scan others
    indices_to_try = [source] + [i for i in range(5) if i != source]
    for idx in indices_to_try:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                if idx != source:
                    print(
                        f"[Data] Webcam index {source} failed, using index {idx} instead."
                    )
                return cap
            cap.release()
    raise RuntimeError("No working webcam found (tried indices 0..4).")


if __name__ == "__main__":
    imgs = download_sample_images()
    print(f"[Data] Sample images ready: {imgs}")
