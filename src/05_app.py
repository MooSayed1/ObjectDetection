"""
Person E - Application / Demo Module
====================================
Responsibility: Glue everything together. Provides 3 modes:
    - image  : detect on a single image
    - video  : detect on every frame of a video file
    - webcam : detect on live webcam feed

Run:
    python 05_app.py --mode image  --source ../data/street.jpg
    python 05_app.py --mode video  --source ../data/sample.mp4
    python 05_app.py --mode webcam
"""

import os
import sys
import cv2
import time
import argparse
from importlib import import_module

# Allow importing modules whose names start with a digit
sys.path.insert(0, os.path.dirname(__file__))
data_mod = import_module("01_data_loader")
model_mod = import_module("02_model")
det_mod = import_module("03_detector")
viz_mod = import_module("04_visualizer")

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def run_image(model, src):
    img = data_mod.load_image(src)
    dets = det_mod.detect(model, img)
    out = viz_mod.draw_detections(img, dets)
    save = os.path.join(OUT_DIR, "image_result.jpg")
    cv2.imwrite(save, out)
    print(f"[App] Detected {len(dets)} objects. Saved -> {save}")
    for d in dets:
        print("   ", d["class"], f"{d['conf']:.2f}")
    return save


def run_video(model, src, show=True):
    cap = data_mod.get_video_capture(src)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    save = os.path.join(OUT_DIR, "video_result.mp4")
    writer = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = det_mod.detect(model, frame)
        out = viz_mod.draw_detections(frame, dets)
        writer.write(out)
        n += 1
        if show:
            cv2.imshow("Object Detection - Video", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()
    dt = time.time() - t0
    print(f"[App] Processed {n} frames in {dt:.1f}s ({n / max(dt, 1e-6):.1f} FPS).")
    print(f"[App] Saved -> {save}")
    return save


def run_webcam(model, cam_index=0):
    cap = data_mod.get_video_capture(cam_index)
    print("[App] Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = det_mod.detect(model, frame)
        out = viz_mod.draw_detections(frame, dets)
        cv2.imshow("Object Detection - Webcam", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["image", "video", "webcam"], default="image")
    p.add_argument(
        "--source", default=None, help="path to image or video (not needed for webcam)"
    )
    p.add_argument("--no-show", action="store_true", help="don't open display windows")
    args = p.parse_args()

    model = model_mod.load_model()

    if args.mode == "image":
        if args.source is None:
            paths = data_mod.download_sample_images()
            args.source = paths[0]
        run_image(model, args.source)

    elif args.mode == "video":
        if args.source is None:
            raise SystemExit("Please provide --source path/to/video.mp4")
        run_video(model, args.source, show=not args.no_show)

    elif args.mode == "webcam":
        run_webcam(model, 0)


if __name__ == "__main__":
    main()
