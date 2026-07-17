# Deployment

Three interchangeable paths — all end with the same server on port 8000.

## A. Plain VM (recommended: Oracle Always Free / GCP / Hetzner)

```bash
# Ubuntu 22.04+, as root once:
adduser --system --group --home /opt/krishidrishti krishi
apt install -y python3-venv nodejs npm caddy

# as the app user / in /opt/krishidrishti:
git clone <your-repo-url> .
python3 -m venv venv && venv/bin/pip install -r requirements.txt
cd frontend && npm ci && npm run build && cd ..

# copy the pieces that are NOT in git:
#   runs/final_classifier_rebuilt/  runs/tempcnn_rebuilt/  runs/stress_lstm/
#   gee_key.json   (Earth Engine service-account key)

sudo cp deploy/krishidrishti.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now krishidrishti
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile   # edit the domain first
sudo systemctl reload caddy
```

Point your domain's DNS A record at the VM — Caddy fetches the HTTPS
certificate automatically. Done.

## B. Docker (any container host, incl. Hugging Face Spaces)

```bash
# from the repo root (models must be present in runs/ first)
docker build -t krishidrishti .
docker run -d -p 8000:8000 \
  -v $PWD/gee_key.json:/app/gee_key.json:ro \
  -v krishi_runs:/app/aoi_runs \
  krishidrishti
```

For HF Spaces: create a Docker Space, use deploy/hf_space_README.md as the
Space's README.md (its front-matter sets app_port: 8000), git-lfs track
`*.joblib`, include runs/ model folders, and add the key file's JSON
content as a Space secret named `GEE_KEY_JSON`. No local Docker needed —
Hugging Face builds the image from the pushed repo.

## C. Your own PC + Cloudflare Tunnel (free, demo-grade)

```powershell
# server side (already works today):
start_servers.cmd            # or Task Scheduler "At startup"
# tunnel side (one-time: winget install Cloudflare.cloudflared):
cloudflared tunnel --url http://127.0.0.1:8000
```

`cloudflared` prints a public https URL that anyone can open. The site is
up while your PC is on.

## Notes

- **Never** commit or bake `gee_key.json` into an image; mount or secret it.
- The service currently authenticates with a personal EE login; switch to
  the service account before exposing it publicly.
- GEE quota is per-project regardless of host; keep the job concurrency cap
  (default 2) and per-IP limits enabled.
