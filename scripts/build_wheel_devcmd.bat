@echo off
setlocal
rem VS 2022 on GHA is Community; local installs may be Professional/Enterprise/BuildTools.
set "VCVARS="
for %%E in (Community Professional Enterprise BuildTools) do (
  if exist "C:\Program Files\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat"
    goto :found_vcvars
  )
)
echo ERROR: Visual Studio 2022 VC tools not found.
exit /b 1

:found_vcvars
call "%VCVARS%"
if exist "C:\Program Files\CMake\bin" (
  set "PATH=C:\Program Files\CMake\bin;%PATH%"
)

cd /d "%~dp0.."

set "PYTHON_EXE=python"
if defined PYTHON set "PYTHON_EXE=%PYTHON%"

set "PS_ARGS=-Python %PYTHON_EXE%"
if /I "%~1"=="-Incremental" set "PS_ARGS=%PS_ARGS% -Incremental"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_wheel_windows.ps1" %PS_ARGS%
exit /b %ERRORLEVEL%
