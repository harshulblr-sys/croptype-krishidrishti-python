@echo off
rem Dev-server launcher: ensures Node.js is on PATH (needed when the parent
rem process started before Node was installed).
set "PATH=C:\Program Files\nodejs;%PATH%"
npm run dev
