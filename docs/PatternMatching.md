# Pattern matching (OpenCvPython)

Python bindings for the same native matchers as **OpenCvSharp** (`OpenCvSharpExtern`).

## Modules

After `pip install` with contrib enabled:

```python
import cv2

# Shape-based (Halcon / PatMax style)
m = cv2.pattern_matching.ShapeBasedMatcher_create()

# Fast pyramid NCC
n = cv2.pattern_matching.FastNCCMatcher_create()

# MatchTool-style NCC (SIMD + optional sub-pixel)
t = cv2.pattern_matching.MatchToolNCCMatcher_create()
```

## Shape matcher quick start

```python
import cv2

template = cv2.imread("part_template.png", cv2.IMREAD_GRAYSCALE)
scene = cv2.imread("scene.png", cv2.IMREAD_GRAYSCALE)

matcher = cv2.pattern_matching.ShapeBasedMatcher_create()
matcher.setTemplate(template, None)  # None = no mask (use full template)
matcher.setAngleRange(-180, 180)
matcher.angleStep = 0  # auto (Halcon-style)
matcher.metric = cv2.pattern_matching.SHAPE_MATCH_USE_POLARITY
matcher.train()

hits = matcher.find(scene, 0.7, 10)
for h in hits:
    print(h.score, h.angle, h.position)
```

## Save / load model

```python
matcher.saveModel("part.shapemodel.yml")
# ...
matcher.loadModel("part.shapemodel.yml")
hits = matcher.find(scene, 0.75, 5)
```

## When to use which matcher

| Matcher | Best for |
|---------|----------|
| `ShapeBasedMatcher` | Outlines, variable lighting, occlusion |
| `FastNCCMatcher` | Texture, stable lighting |
| `MatchToolNCCMatcher` | Same as Fast NCC + sub-pixel refinement |

Full C# API reference and tuning tables: see OpenCvSharp [ShapeBasedMatcher.md](https://github.com/newocean-group/OpenCvSharp/blob/main/docs/ShapeBasedMatcher.md).

## Sync native code from OpenCvSharp

```powershell
.\scripts\sync_from_opencvsharp.ps1 -OpenCvSharpRoot C:\path\to\OpenCvSharp
```
