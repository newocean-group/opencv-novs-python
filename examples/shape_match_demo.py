"""Shape-based match demo — requires opencv-novs-python wheel with opencv_novs."""
import sys
from pathlib import Path

import cv2
import numpy as np

HIT_COLORS = (
    (0, 255, 0),
    (0, 200, 255),
    (255, 128, 0),
    (255, 0, 200),
    (200, 255, 0),
)


def _scene_bgr(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _template_polygon(hit, template_w: int, template_h: int) -> np.ndarray:
    """Rotated rectangle (4 corners) — match position is template centre."""
    x, y = hit.position
    scale_r = float(getattr(hit, "scaleR", 1.0))
    scale_c = float(getattr(hit, "scaleC", 1.0))
    size = (float(template_w * scale_c), float(template_h * scale_r))
    rect = ((float(x), float(y)), size, float(hit.angle))
    return np.int32(cv2.boxPoints(rect))


def _draw_hit(out, hit, index: int, template_w: int, template_h: int):
    color = HIT_COLORS[index % len(HIT_COLORS)]
    poly = _template_polygon(hit, template_w, template_h)
    cv2.polylines(out, [poly], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

    x, y = hit.position
    cx, cy = int(round(x)), int(round(y))
    cv2.drawMarker(out, (cx, cy), color, markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
    label = f"#{index} s={hit.score:.3f} a={hit.angle:.1f}"
    cv2.putText(
        out, label, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
    )


def _result_path(scene_path: str) -> Path:
    scene = Path(scene_path)
    return Path.cwd() / f"{scene.stem}_shape_match{scene.suffix or '.jpg'}"


def main():
    if len(sys.argv) < 3:
        print("Usage: shape_match_demo.py <template.png> <scene.png>")
        return 1

    template = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    scene_vis = _scene_bgr(sys.argv[2])
    scene_gray = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
    if template is None or scene_gray is None or scene_vis is None:
        print("Failed to read images")
        return 1

    th, tw = template.shape[:2]

    pm = cv2.pattern_matching
    matcher = pm.ShapeBasedMatcher_create()
    matcher.setTemplate(template, None)
    matcher.setAngleRange(-10, 10)
    matcher.train()

    hits = matcher.find(scene_gray, 0.5, 5)
    print(f"Found {len(hits)} hit(s)")
    for i, h in enumerate(hits):
        x, y = h.position
        print(f"  [{i}] score={h.score:.3f} angle={h.angle:.2f} pos=({x:.1f},{y:.1f})")
        _draw_hit(scene_vis, h, i, tw, th)

    out_path = _result_path(sys.argv[2])
    if not cv2.imwrite(str(out_path), scene_vis):
        print(f"Failed to write {out_path}")
        return 1

    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
