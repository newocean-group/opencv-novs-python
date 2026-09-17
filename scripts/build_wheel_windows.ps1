# Build opencv-contrib-python wheel with New Ocean pattern_matching module (Windows x64).
param(
    [string]$Python = "python",
    [switch]$Headless,
    [switch]$Incremental
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

& $Python -m pip install --upgrade pip scikit-build wheel
& $Python -m pip install "numpy==2.0.2"
# Stale _skbuild caches pip temp paths for NumPy headers; wipe before wheel build.
if (-not $Incremental -and (Test-Path "_skbuild")) {
    Write-Host "Removing stale _skbuild (NumPy header paths from prior pip env)..."
    Remove-Item -Recurse -Force "_skbuild"
}
elseif ($Incremental) {
    $cmakeCache = "_skbuild\win-amd64-3.12\cmake-build\CMakeCache.txt"
    if (Test-Path $cmakeCache) {
        Write-Host "Incremental: dropping CMakeCache to refresh NumPy/bindings paths..."
        Remove-Item -Force $cmakeCache
    }
}
& $Python -m pip wheel . -w dist -v --no-deps

Write-Host "`nWheel(s) in: $root\dist"
