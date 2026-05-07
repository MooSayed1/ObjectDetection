"""
Person F - Metrics & Evaluation Module
=======================================
Responsibility: Measure how well / how fast the detector runs.
Exam topics covered: Model evaluation, FPS measurement, class statistics.

Functions:
    - MetricsTracker class: keeps running stats across frames.
    - print_report():       prints a nice summary table at the end.
    - save_report(path):    writes the report to a text file.
"""

import time
from collections import Counter


class MetricsTracker:
    """Tracks performance + detection statistics across many frames."""

    def __init__(self):
        self.frame_count = 0
        self.total_inference_t = 0.0
        self.class_counts = Counter()
        self.confidence_sum = 0.0
        self.detection_count = 0
        self.start_time = time.time()

    def start_frame(self):
        """Call right before running detection on a frame."""
        self._t0 = time.time()

    def end_frame(self, detections):
        """Call right after detection. Updates all stats."""
        dt = time.time() - self._t0
        self.total_inference_t += dt
        self.frame_count += 1
        for d in detections:
            self.class_counts[d["class"]] += 1
            self.confidence_sum += d["conf"]
            self.detection_count += 1

    @property
    def avg_fps(self):
        if self.total_inference_t == 0:
            return 0.0
        return self.frame_count / self.total_inference_t

    @property
    def avg_confidence(self):
        if self.detection_count == 0:
            return 0.0
        return self.confidence_sum / self.detection_count

    @property
    def avg_objects_per_frame(self):
        if self.frame_count == 0:
            return 0.0
        return self.detection_count / self.frame_count

    def report_lines(self):
        wall = time.time() - self.start_time
        lines = [
            "=" * 50,
            "  OBJECT DETECTION - PERFORMANCE REPORT",
            "=" * 50,
            f"  Frames processed     : {self.frame_count}",
            f"  Total wall time      : {wall:.2f} s",
            f"  Total inference time : {self.total_inference_t:.2f} s",
            f"  Average FPS          : {self.avg_fps:.2f}",
            f"  Total detections     : {self.detection_count}",
            f"  Avg objects / frame  : {self.avg_objects_per_frame:.2f}",
            f"  Avg confidence       : {self.avg_confidence:.3f}",
            "-" * 50,
            "  Detections per class:",
        ]
        if not self.class_counts:
            lines.append("    (none)")
        else:
            for cls, cnt in self.class_counts.most_common():
                lines.append(f"    {cls:<20s} {cnt}")
        lines.append("=" * 50)
        return lines

    def print_report(self):
        for line in self.report_lines():
            print(line)

    def save_report(self, path):
        with open(path, "w") as f:
            f.write("\n".join(self.report_lines()))
        print(f"[Metrics] Report saved -> {path}")


if __name__ == "__main__":
    # Tiny self-test
    m = MetricsTracker()
    fake = [{"class": "person", "conf": 0.9}, {"class": "car", "conf": 0.7}]
    for _ in range(3):
        m.start_frame()
        time.sleep(0.05)
        m.end_frame(fake)
    m.print_report()
