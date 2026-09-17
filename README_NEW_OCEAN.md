# opencv-novs-python (New Ocean fork)

Fork of [opencv-python](https://github.com/opencv/opencv-python) with **industrial pattern matching** from [OpenCvSharp](https://github.com/newocean-group/OpenCvSharp) (New Ocean fork of [shimat/opencvsharp](https://github.com/shimat/opencvsharp)).

| Repo | Role |
|------|------|
| [opencv-novs-python](https://github.com/newocean-group/opencv-novs-python) | Packaging + wheel build (this repo) |
| [opencv-novs](https://github.com/newocean-group/opencv-novs) | C++ `pattern_matching` OpenCV module (submodule) |

Same algorithms as OpenCvSharp `OpenCvSharpExtern`:

| Matcher | OpenCvSharp (C#) | OpenCvPython (after build) |
|---------|------------------|----------------------------|
| Pyramid NCC | `FastNCCMatcher` | `cv2.pattern_matching.FastNCCMatcher_create()` |
| MatchTool NCC | `MatchToolNCCMatcher` | `cv2.pattern_matching.MatchToolNCCMatcher_create()` |
| Shape-based (Halcon-style) | `ShapeBasedMatcher` | `cv2.pattern_matching.ShapeBasedMatcher_create()` |

Native sources live in `pattern_matching/` submodule (~9.4k lines C++). Sync from OpenCvSharp:

```powershell
.\scripts\sync_from_opencvsharp.ps1
```

## Build (contrib + pattern_matching)

Requires: Python 3.9+, CMake, Visual Studio Build Tools (Windows) or gcc/clang (Linux), Git.

```powershell
git submodule update --init --recursive opencv opencv_contrib pattern_matching
pip install --upgrade pip scikit-build numpy
# contrib.enabled in repo root enables opencv-contrib modules + pattern_matching
pip install . -v
```

Or explicitly:

```powershell
$env:ENABLE_CONTRIB = "1"
pip install . -v
```

Wheel name follows upstream (`opencv-contrib-python` unless `OPENCV_PYTHON_PACKAGE_NAME` is set).

## Quick example

```python
import cv2

matcher = cv2.pattern_matching.ShapeBasedMatcher_create()
matcher.setTemplate(template_bgr)
matcher.train()
hits = matcher.find(scene_bgr, 0.5, 1)
for h in hits:
    print(h.score, h.angle, h.position)
```

See [docs/PatternMatching.md](docs/PatternMatching.md) for tuning (greediness, MinContrast, pyramid, Halcon parity notes).

## Repository layout

```
OpenCvPython/                 # this fork (opencv-python CI + packaging)
  opencv/                     # submodule — upstream OpenCV
  opencv_contrib/             # submodule — upstream contrib
  pattern_matching/           # submodule — New Ocean pattern matcher (opencv-novs)
  contrib.enabled             # default ON: build contrib + pattern_matching
  scripts/sync_from_opencvsharp.ps1
```

## Differences vs OpenCvSharp

- **No RSA device license** in Python build (`opencvsharp_license_guard` stub always active).
- **C# utilities** (`ShapeBasedMatcherUtilities.cs` belt ROI, two-pass score) are app-layer — port to Python as needed; core find/train API matches.
- **Package**: `import cv2` + `cv2.pattern_matching` submodule (OpenCV contrib binding style).

## Upstream sync

1. Merge/rebase from `opencv/opencv-python` `4.x` branch.
2. Run `sync_from_opencvsharp.ps1` after OpenCvSharp native changes.
3. Rebuild wheels; run regression tests (see OpenCvSharp `smoke_test/` — Halcon compare harness is C#/Halcon-specific).

## License

Follows opencv-python / OpenCV licensing. Pattern matching C++ retains New Ocean / OpenCvSharp project license terms — see `LICENSE.txt` and OpenCvSharp repo.
