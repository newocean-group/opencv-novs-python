@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
set "PATH=C:\Program Files\CMake\bin;%PATH%"
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_wheel_windows.ps1" -Python "C:\Users\thuon\AppData\Local\Programs\Python\Python312\python.exe"
exit /b %ERRORLEVEL%
