# Sync pattern_matching native sources from OpenCvSharp → pattern_matching submodule.
# After sync: commit/push in pattern_matching, then bump submodule ref in parent.
param(
    [string]$OpenCvSharpRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\OpenCvSharp")).Path,
    [string]$OpenCvPythonRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)
$ErrorActionPreference = "Stop"
$srcDir = Join-Path $OpenCvSharpRoot "src\OpenCvSharpExtern"
$dstSrc = Join-Path $OpenCvPythonRoot "pattern_matching\src"
$dstInc = Join-Path $OpenCvPythonRoot "pattern_matching\include\opencv2\pattern_matching"
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
Write-Host "Synced pattern_matching sources from $srcDir"
Write-Host "Next: cd pattern_matching && git commit -am '...' && git push"
Write-Host "Then: cd .. && git add pattern_matching && git commit -m 'Bump pattern_matching submodule'"
