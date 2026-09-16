# pattern_matching (OpenCV contrib extra module)

Native C++ ported from `OpenCvSharp/src/OpenCvSharpExtern`:

- `fast_ncc_matcher.*` — pyramid NCC
- `match_tool_ncc_matcher.*` — MatchTool-style NCC
- `shape_based_matcher.*` — shape-based matching (~5.5k lines)

Python: `cv2.pattern_matching.*` after building OpenCvPython with `ENABLE_CONTRIB=1`.

Update sources: `../../scripts/sync_from_opencvsharp.ps1`
