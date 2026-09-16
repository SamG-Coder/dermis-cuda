@echo off
setlocal
cd /d "%~dp0"
where node >nul 2>nul
if errorlevel 1 (
  echo Node.js 20 or newer is required. Install Node.js, then run START.bat again.
  pause
  exit /b 1
)
if not defined PORT set PORT=8080
echo Open http://localhost:%PORT% in a WebGPU-enabled browser.
echo Keep this window open while using the portrait.
node scripts\serve.mjs
pause
