@echo off
rem Serve KrishiDrishti to the public internet via a Cloudflare quick tunnel.
rem Prereqs (one-time):
rem   1. cloudflared installed  (winget install --id Cloudflare.cloudflared)
rem   2. frontend built          (cd frontend ^&^& npm run build)
rem Double-click this file. Two windows open; the tunnel window prints a
rem public https://<random>.trycloudflare.com URL. Close the windows to stop.
set "PATH=C:\Program Files\nodejs;%PATH%"
start "KrishiDrishti server :8000" cmd /k "cd /d %~dp0 && python aoi_server.py"
echo Waiting for the server to come up...
timeout /t 4 >nul
start "Cloudflare tunnel" cmd /k "cloudflared tunnel --url http://localhost:8000"
echo.
echo The "Cloudflare tunnel" window prints your public URL (trycloudflare.com).
echo Share that link. It stays valid while these two windows are open.
