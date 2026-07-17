# Stable public URL via a named Cloudflare Tunnel (from your own PC)

The quick tunnel (`start_public.cmd`) is instant but its URL rotates and it
stops when you close the window. This upgrades to a **fixed hostname** that
**auto-starts on boot** — you never run a script manually again. It needs a
domain on a (free) Cloudflare account.

## 1. Get a domain (free for students)
GitHub Student Pack → **Namecheap** offer → register a free `.me` domain
(e.g. `krishidrishti.me`). Any domain works; this one's free.

## 2. Add the domain to Cloudflare (free)
1. Sign up at **dash.cloudflare.com** → **Add a site** → enter your domain.
2. Cloudflare shows you **two nameservers**. In Namecheap → your domain →
   **Nameservers → Custom DNS** → paste both → save.
3. Wait for the "site is active" email (minutes to a few hours).

## 3. Create the tunnel (Zero Trust dashboard — no config files)
1. **one.dash.cloudflare.com** → **Networks → Tunnels → Create a tunnel** →
   type **Cloudflared** → name it `krishidrishti`.
2. It shows an install command containing a long token. On your PC, open
   **PowerShell as Administrator** and run it — it looks like:
   ```
   cloudflared.exe service install eyJ...<long token>...
   ```
   This installs cloudflared as a **Windows service** (auto-starts on boot)
   and connects it. The dashboard should flip the connector to "Healthy".
3. Still in the tunnel setup → **Public Hostnames → Add a public hostname**:
   - Subdomain: `app` (or blank for the root)
   - Domain: your domain
   - Service: **HTTP**  ·  URL: **localhost:8000**
   - Save.

Your site is now at `https://app.yourdomain.me` — a fixed URL, HTTPS
handled by Cloudflare, tunnel restarting on boot.

## 4. Auto-start the server too (Task Scheduler)
The tunnel service is up on boot, but it needs the app running behind it.
Make `aoi_server.py` start automatically:

1. **Task Scheduler → Create Task** (not "Basic Task").
2. General: name `KrishiDrishti server`; select **Run only when user is
   logged on** (simplest).
3. Triggers → New → **At log on** (of your user).
4. Actions → New → Program/script: `cmd.exe`; Add arguments:
   `/c "cd /d C:\Users\harsh\OneDrive\Desktop\Harshul(1)\ISRO_Hackathon && python aoi_server.py"`
5. OK. It now launches on every login; reboot to test.

## Result
Reboot the PC → tunnel service + server both come back → the same fixed
URL works, no manual steps. (The site is still only up while the PC is
powered on — that's inherent to hosting from your own machine.)

## Notes
- The server binds 127.0.0.1:8000 (default) — correct: only the local
  tunnel reaches it, nothing else is exposed.
- To stop publishing: disable the Task Scheduler task and, in the Cloudflare
  dashboard, pause or delete the tunnel.
