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


def _template_rect_polygon(hit, template_w: int, template_h: int) -> np.ndarray:
    """Place the template rectangle on the scene at the matched pose.

    Template geometry is an axis-aligned W×H rectangle in model space.
    Native find() reports ``position`` as the rectangle centre, plus angle
    and optional anisotropic scale — same convention as OpenCvSharp
    ``RotatedRect(hit.Position, Size2f(W*ScaleC, H*ScaleR), hit.Angle)``.
    """
    x, y = hit.position
    scale_r = float(getattr(hit, "scaleR", 1.0))
    scale_c = float(getattr(hit, "scaleC", 1.0))
    size = (float(template_w * scale_c), float(template_h * scale_r))
    center = (float(x), float(y))
    angle = float(hit.angle)
    return np.int32(np.round(cv2.boxPoints((center, size, angle))))


def _label_pos_above_polygon(poly: np.ndarray, text: str, gap: int = 10):
    """Place label fully above the polygon so it does not overlap the border."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.85
    thickness = 2
    pad = 4
    (tw_label, th_label), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x_min = int(poly[:, 0].min())
    x_max = int(poly[:, 0].max())
    y_min = int(poly[:, 1].min())

    tx = int((x_min + x_max - tw_label) / 2)
    # Bottom of label background sits above the topmost polygon edge.
    ty = y_min - gap - baseline - pad
    return tx, ty, font, font_scale, thickness, tw_label, th_label, baseline, pad


def _draw_label(out, text: str, poly: np.ndarray, color):
    tx, ty, font, font_scale, thickness, tw_label, th_label, baseline, pad = (
        _label_pos_above_polygon(poly, text)
    )
    cv2.rectangle(
        out,
        (tx - pad, ty - th_label - pad),
        (tx + tw_label + pad, ty + baseline + pad),
        (0, 0, 0),
        -1,
    )
    cv2.putText(out, text, (tx, ty), font, font_scale, color, thickness, cv2.LINE_AA)


def _draw_hit(out, hit, index: int, template_w: int, template_h: int):
    color = HIT_COLORS[index % len(HIT_COLORS)]
    poly = _template_rect_polygon(hit, template_w, template_h)

    x, y = hit.position
    cx, cy = int(round(x)), int(round(y))
    cv2.drawMarker(out, (cx, cy), color, markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)

    label = f"#{index} s={hit.score:.3f} a={hit.angle:.1f}"
    _draw_label(out, label, poly, color)

    # Draw polygon after label so the border is never covered by the text background.
    cv2.polylines(out, [poly], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)


def _result_path(scene_path: str) -> Path:
    scene = Path(scene_path)
    return Path.cwd() / f"{scene.stem}_shape_match{scene.suffix or '.jpg'}"


def main():
    if len(sys.argv) < 3:
        print("Usage: shape_match_demo.py <template.png> <scene.png> [maxTargets]")
        return 1

    max_targets = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    template = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    scene_vis = _scene_bgr(sys.argv[2])
    scene_gray = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
    if template is None or scene_gray is None or scene_vis is None:
        print("Failed to read images")
        return 1

    # Template is a concrete axis-aligned rectangle (polygon with 4 corners).
    template_h, template_w = template.shape[:2]

    pm = cv2.pattern_matching
    matcher = pm.ShapeBasedMatcher_create()
    matcher.setTemplate(template, None)
    matcher.setAngleRange(-10, 10)
    matcher.train()

    hits = matcher.find(scene_gray, 0.5, max_targets)
    print(f"Found {len(hits)} hit(s) (maxTargets={max_targets})")
    for i, h in enumerate(hits):
        x, y = h.position
        print(f"  [{i}] score={h.score:.3f} angle={h.angle:.2f} pos=({x:.1f},{y:.1f})")
        _draw_hit(scene_vis, h, i, template_w, template_h)

    out_path = _result_path(sys.argv[2])
    if not cv2.imwrite(str(out_path), scene_vis):
        print(f"Failed to write {out_path}")
        return 1

    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
