@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist ".venv\Scripts\python.exe" goto not_ready
".venv\Scripts\python.exe" test_demo.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" demo.py --schema cslist_schema.json --search-only
if errorlevel 1 goto failed
echo Local checks passed. No API request was made.
pause
exit /b 0
:not_ready
echo Run setup.bat first.
pause
exit /b 1
:failed
echo Check failed. Read the error above.
pause
exit /b 1
