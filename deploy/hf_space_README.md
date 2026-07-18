---
title: KrishiDrishti
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
---

# KrishiDrishti — Crop Map & Irrigation Advisory

Draw a box over northern India → Sentinel-1/2 crop map, moisture stress,
8-day water deficit & irrigation advisory.

This Space runs the FastAPI service in `Dockerfile`. The Earth Engine
service-account key is supplied via the Space secret `GEE_KEY_JSON`
(the full JSON content of the key file).
