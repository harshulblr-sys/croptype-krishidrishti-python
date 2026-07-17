# KrishiDrishti AOI service — container image (VM, Hugging Face Spaces, etc.)
#
# Build from the repo root (the .dockerignore keeps datasets out):
#   docker build -t krishidrishti .
# (Lives at the repo root because Hugging Face Docker Spaces require it here.)
# Run:
#   docker run -p 8000:8000 -e PORT=8000 \
#     -v /path/to/gee_key.json:/app/gee_key.json:ro \
#     -v krishi_runs:/app/aoi_runs \
#     krishidrishti
#
# Prerequisite: the trained model artifacts must exist in the build context:
#   runs/final_classifier_rebuilt/  runs/tempcnn_rebuilt/  runs/stress_lstm/
# (they are gitignored — copy them in before building).

# ---- stage 1: build the React frontend ----
FROM node:20-slim AS webbuild
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- stage 2: Python runtime ----
FROM python:3.11-slim
WORKDIR /app

# libgomp1: LightGBM's OpenMP runtime (rasterio/torch wheels are self-contained)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch keeps the image ~5 GB smaller than the CUDA default
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY *.py dashboard_template.html ./
COPY runs/ runs/
COPY --from=webbuild /web/dist frontend/dist

# Earth Engine service-account key: mount at /app/gee_key.json (never bake
# credentials into the image).
ENV PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8000
EXPOSE 8000

CMD ["python", "aoi_server.py"]
