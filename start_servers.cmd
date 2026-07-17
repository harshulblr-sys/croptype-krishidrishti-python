@echo off
rem Start the PS-6 AOI service (production mode).
rem One server does everything: FastAPI on :8000 serves the API, the built
rem React frontend (frontend/dist) and all job outputs.
rem Double-click this file; close the window (or Ctrl+C) to stop.
rem
rem Frontend development with hot reload instead: run frontend\dev.cmd
rem (Vite on :5173, proxies /api to :8000).
start "PS6 server :8000" cmd /k "cd /d %~dp0 && python aoi_server.py"
echo Site + API -> http://127.0.0.1:8000
