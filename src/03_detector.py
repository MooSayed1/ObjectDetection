"""
Person C - Detection / Inference Module
=======================================
Responsibility: Run the model on an image/frame and return detections.
Exam topics covered: Inference pipeline, confidence thresholding.

A "detection" is a dict:
    {
        "box":   (x1, y1, x2, y2),   # bounding box in pixels
        "class": "person",
        "conf":  0.87
    }
"""


def detect(model, frame, conf_threshold: float = 0.4):
    """
    Run YOLO on a single BGR frame.
    Returns: list of detection dicts.
    """
    results = model.predict(frame, conf=conf_threshold, verbose=False)
    detections = []
    if not results:
        return detections

    r = results[0]
    names = r.names
    if r.boxes is None:
        return detections

    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detections.append(
            {
                "box": (x1, y1, x2, y2),
                "class": names[cls_id],
                "conf": conf,
            }
        )
    return detections


if __name__ == "__main__":
    import cv2, sys, os

    sys.path.insert(0, os.path.dirname(__file__))
    from importlib import import_module

    model_mod = import_module("02_model")
    data_mod = import_module("01_data_loader")

    model = model_mod.load_model()
    paths = data_mod.download_sample_images()
    img = data_mod.load_image(paths[0])
    dets = detect(model, img)
    print(f"[Detector] Found {len(dets)} objects:")
    for d in dets:
        print(" ", d)
