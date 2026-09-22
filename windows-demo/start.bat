@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if exist ".venv\Scripts\python.exe" goto ready
call setup.bat
if errorlevel 1 exit /b 1
:ready
".venv\Scripts\python.exe" demo.py --ui
if errorlevel 1 pause
