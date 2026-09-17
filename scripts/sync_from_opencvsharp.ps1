# Sync opencv_novs native sources from OpenCvSharp → opencv_novs submodule.
# After sync: commit/push in opencv_novs, then bump submodule ref in parent.
param(
    [string]$OpenCvSharpRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\OpenCvSharp")).Path,
    [string]$OpenCvPythonRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)
$ErrorActionPreference = "Stop"
$srcDir = Join-Path $OpenCvSharpRoot "src\OpenCvSharpExtern"
$dstSrc = Join-Path $OpenCvPythonRoot "opencv_novs\src"
$dstInc = Join-Path $OpenCvPythonRoot "opencv_novs\include\opencv2\pattern_matching"
$files = @(
    "fast_ncc_matcher.cpp", "fast_ncc_matcher.h",
    "match_tool_ncc_matcher.cpp", "match_tool_ncc_matcher.h",
    "shape_based_matcher.cpp", "shape_based_matcher.h"
)
foreach ($f in $files) {
    Copy-Item (Join-Path $srcDir $f) $dstSrc -Force
    if ($f -like "*.h") {
        $hpp = $f -replace '\.h$','.hpp'
        Copy-Item (Join-Path $srcDir $f) (Join-Path $dstInc $hpp) -Force
    }
}
# Keep OpenCvPython license stub (do not overwrite)
Write-Host "Synced opencv_novs sources from $srcDir"
Write-Host "Next: cd opencv_novs && git commit -am '...' && git push"
Write-Host "Then: cd .. && git add opencv_novs && git commit -m 'Bump opencv_novs submodule'"
