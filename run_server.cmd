@echo off
rem Launcher for Task Scheduler: runs the KrishiDrishti server in the
rem foreground (so the task stays "Running"). Self-locates to the project
rem folder, logs to server.log. Pair with Tailscale Funnel for a public URL.
cd /d "%~dp0"
python aoi_server.py >> server.log 2>&1
