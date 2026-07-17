# Fixed public URL via Tailscale Funnel (from your own PC)

Gives a stable `https://<machine>.<tailnet>.ts.net` URL — free, no domain to
buy, no interstitial page, HTTPS handled for you. Tailscale runs as a
Windows service (auto-starts on boot) and the `--bg` funnel config persists
across reboots, so once set up it stays up whenever the PC is on.

## 1. Install Tailscale
`winget install tailscale.tailscale`  (or download from tailscale.com/download)
Sign in with a Google/GitHub/Microsoft account — a free personal *tailnet*.

## 2. Enable Funnel (one-time, per tailnet)
Funnel is off by default. The easiest path: just run the command in step 3 —
if Funnel isn't enabled yet, Tailscale prints a URL; open it to enable
Funnel + HTTPS for your tailnet in the admin console, then re-run.

(Manual equivalent, admin console at login.tailscale.com:
 DNS → **Enable HTTPS**; Access controls → add
 `"nodeAttrs": [{"target": ["autogroup:member"], "attr": ["funnel"]}]`.)

## 3. Expose the server
With `aoi_server.py` running on :8000:
```powershell
tailscale funnel --bg 8000
```
`--bg` runs it in the background and persists it. Then get your public URL:
```powershell
tailscale funnel status
```
It shows something like `https://harsh-pc.tailabc123.ts.net` → that's your
fixed link. Open it; the site is live.

To stop publishing later: `tailscale funnel --https=443 off`

## 4. Auto-start the server (so it survives reboots too)
Tailscale + the funnel config already restart on boot; make the app do the
same via Task Scheduler:

1. **Task Scheduler → Create Task**.
2. General: name `KrishiDrishti server`; **Run only when user is logged on**.
3. Triggers → New → **At log on**.
4. Actions → New → `cmd.exe`, arguments:
   `/c "cd /d C:\Users\harsh\OneDrive\Desktop\Harshul(1)\ISRO_Hackathon && python aoi_server.py"`
5. OK. Reboot to confirm the URL comes back on its own.

## Notes
- Funnel only serves on ports 443 / 8443 / 10000 publicly; `funnel 8000`
  maps the default (443) to your local :8000 — nothing to configure.
- The URL is long/unmemorable but permanent. For a prettier URL you'd need
  your own domain (see cloudflare_tunnel.md).
- Still only reachable while the PC is powered on — inherent to self-hosting.
- Job submissions are non-blocking (the client polls), so there are no
  long-held requests for the funnel to time out.
