# Build opencv-contrib-python wheel with New Ocean pattern_matching module (Windows x64).
param(
    [string]$Python = "python",
    [switch]$Headless
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

if (-not (Test-Path "opencv\CMakeLists.txt")) {
    Write-Host "Initializing submodules..."
    git submodule update --init --depth 1 opencv opencv_contrib
}

Write-Host "OpenCV version:"
Select-String "CV_VERSION_MAJOR|CV_VERSION_MINOR" opencv\modules\core\include\opencv2\core\version.hpp | Select-Object -First 2

$env:ENABLE_CONTRIB = "1"
if ($Headless) { $env:ENABLE_HEADLESS = "1" }
"1" | Set-Content contrib.enabled -NoNewline

& $Python -m pip install --upgrade pip scikit-build numpy wheel
& $Python -m pip wheel . -w dist -v --no-deps

Write-Host "`nWheel(s) in: $root\dist"
