@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
py -3.14 --version
if errorlevel 1 goto no_python
py -3.14 -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" test_demo.py
if errorlevel 1 goto failed
echo Setup complete. Open start.bat to use the CS RAG menu.
pause
exit /b 0
:no_python
echo Install Python 3.14 x64 with the Python launcher, then retry.
echo https://www.python.org/downloads/windows/
pause
exit /b 1
:failed
echo Setup failed. Read the error above. See README.txt.
pause
exit /b 1
