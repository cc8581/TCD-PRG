@echo off
setlocal
chcp 65001 >nul
set "PYTHON=D:\Anaconda\install\python.exe"
set "LAUNCHER=D:\pycharm\Project\TCD-PRG\scripts\precompute_launcher_zh.py"
set "TCD_PRG_CACHE_DIR=F:\TCD-PRG-observations"
if not exist "%PYTHON%" (
  echo ERROR: Python not found.
  pause
  exit /b 1
)
"%PYTHON%" "%LAUNCHER%"
exit /b %ERRORLEVEL%
