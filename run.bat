@echo off
setlocal
cd /d "%~dp0"

remark ------------------------------------------------------------------
remark  First run: create a private environment (.venv) and install deps.
remark  Needs: Python 3.10 - 3.12 on PATH (see README.md, "NEW MACHINE").
remark  Keep this file plain ASCII: cmd chokes on non-ASCII in .bat files.
remark ------------------------------------------------------------------

if exist ".venv\Scripts\python.exe" goto havevenv

echo [setup] .venv not found - creating it now (one time only).
echo [setup] Downloading packages may take a few minutes. Please wait.
echo.
python -m venv .venv
if errorlevel 1 goto novpython
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto novpip
echo [setup] Done. Starting...
echo.

:havevenv
".venv\Scripts\python.exe" launcher.py %*
if errorlevel 1 pause
exit /b %errorlevel%

:novpython
echo [setup] FAILED: "python" not found, or venv could not be created.
echo.
echo [setup] Install Python 3.10 - 3.12 from
echo [setup]     https://www.python.org/downloads/
echo [setup] and tick "Add python.exe to PATH" during install,
echo [setup] then double-click run.bat again.
echo [setup] Note: the Microsoft Store "python" stub is not enough.
pause
exit /b 1

:novpip
echo [setup] FAILED: package install hit an error (network / proxy / antivirus?).
echo [setup] Retry by hand in this folder:
echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
pause
exit /b 1
