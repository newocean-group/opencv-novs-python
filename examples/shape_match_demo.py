"""Minimal shape-based match demo — requires opencv-novs-python build with opencv_novs."""
import sys
import cv2


def main():
    if len(sys.argv) < 3:
        print("Usage: shape_match_demo.py <template.png> <scene.png>")
        return 1

    template = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    scene = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
    if template is None or scene is None:
        print("Failed to read images")
        return 1

    pm = cv2.pattern_matching
    matcher = pm.ShapeBasedMatcher_create()
    matcher.setTemplate(template)
    matcher.setAngleRange(-10, 10)
    matcher.train()

    hits = matcher.find(scene, 0.5, 5)
    print(f"Found {len(hits)} hit(s)")
    for i, h in enumerate(hits):
        print(f"  [{i}] score={h.score:.3f} angle={h.angle:.2f} pos=({h.position.x:.1f},{h.position.y:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
