@echo off
rem Start the PS-6 AOI service: FastAPI backend (:8000) + Vite frontend (:5173).
rem Double-click this file, or run it from any terminal. Each server opens in
rem its own window; close the windows (or Ctrl+C in them) to stop.
start "PS6 backend :8000" cmd /k "cd /d %~dp0 && python aoi_server.py"
start "PS6 frontend :5173" cmd /k "cd /d %~dp0frontend && dev.cmd"
echo Backend  -> http://127.0.0.1:8000
echo Frontend -> http://localhost:5173
