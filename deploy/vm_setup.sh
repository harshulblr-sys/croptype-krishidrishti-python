#!/usr/bin/env bash
# One-shot setup for an Ubuntu VM — DigitalOcean, Google Cloud, Oracle, etc.
# (x86 or ARM; installs the right wheels either way.)
#
# Prerequisites, already done before you run this:
#   1. The repo is cloned into the CURRENT directory (git clone ...).
#   2. You have placed the two untracked pieces INSIDE this directory:
#        - gee_key.json                      (Earth Engine service-account key)
#        - runs/final_classifier_rebuilt/    (+ tempcnn_rebuilt/ + stress_lstm/)
#
# Usage (from inside the cloned repo):
#   sudo bash deploy/vm_setup.sh <public-hostname>
#   e.g.  sudo bash deploy/vm_setup.sh 140-238-1-2.nip.io   (a free hostname)
#   e.g.  sudo bash deploy/vm_setup.sh krishidrishti.me     (your own domain)
#   ('<dashed-ip>.nip.io' resolves to your IP automatically, so Caddy can
#    fetch a real HTTPS certificate without buying a domain.)
#
# Re-running is safe.
set -euo pipefail

HOST="${1:?Pass your public hostname, e.g. 140-238-1-2.nip.io}"
APP=/opt/krishidrishti
SRC="$(pwd)"

echo "== 1/6  system packages =="
apt-get update -y
apt-get install -y python3-venv python3-pip nodejs npm git debian-keyring \
                   debian-archive-keyring apt-transport-https curl
# Caddy (official apt repo)
if ! command -v caddy >/dev/null; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -y && apt-get install -y caddy
fi

echo "== 2/6  app user + files =="
id krishi &>/dev/null || adduser --system --group --home "$APP" krishi
mkdir -p "$APP"
# copy the repo (incl. gee_key.json and runs/) into place
cp -r "$SRC"/. "$APP"/
chown -R krishi:krishi "$APP"

echo "== 3/6  python venv + deps =="
sudo -u krishi python3 -m venv "$APP/venv"
sudo -u krishi "$APP/venv/bin/pip" install --upgrade pip
sudo -u krishi "$APP/venv/bin/pip" install -r "$APP/requirements.txt"

echo "== 4/6  build frontend =="
cd "$APP/frontend"
sudo -u krishi npm ci
sudo -u krishi npm run build
cd "$APP"

echo "== 5/6  OS firewall (Oracle Ubuntu blocks 80/443; a no-op elsewhere) =="
# insert ACCEPT at the top of INPUT so it precedes any REJECT rule (Oracle
# images ship one; DigitalOcean/GCP don't, so this is simply harmless there)
iptables -I INPUT -p tcp --dport 80  -j ACCEPT
iptables -I INPUT -p tcp --dport 443 -j ACCEPT
netfilter-persistent save 2>/dev/null || \
  (apt-get install -y iptables-persistent && netfilter-persistent save) || \
  echo "  (no netfilter-persistent; rules apply for this boot)"

echo "== 6/6  systemd service + Caddy =="
cp "$APP/deploy/krishidrishti.service" /etc/systemd/system/
# point the service at the on-disk key
sed -i 's|# Environment=GEE_KEY_FILE=.*|Environment=GEE_KEY_FILE='"$APP"'/gee_key.json|' \
    /etc/systemd/system/krishidrishti.service
systemctl daemon-reload
systemctl enable --now krishidrishti

# Caddy site with the real hostname
sed "s/krishidrishti.example.com/$HOST/" "$APP/deploy/Caddyfile" > /etc/caddy/Caddyfile
systemctl reload caddy

echo
echo "DONE.  https://$HOST  should be live within a minute (first request"
echo "waits while Caddy fetches the TLS certificate)."
echo "Logs:   journalctl -u krishidrishti -f"
