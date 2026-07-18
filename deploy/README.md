# Deployment

The public site runs from your own PC, exposed with a free **Tailscale
Funnel** — a stable `https://<machine>.<tailnet>.ts.net` URL, no domain to
buy, no hosting cost, HTTPS handled for you. Full step-by-step (install,
enable Funnel, auto-start on boot): **[tailscale_funnel.md](tailscale_funnel.md)**.

## Your own PC + Tailscale Funnel

```powershell
# 1. Server — serves the built frontend + API on :8000
start_servers.cmd            # or auto-start via Task Scheduler (see below)

# 2. Tunnel — one-time: winget install tailscale.tailscale, then log in
tailscale funnel --bg 8000
tailscale funnel status      # prints your public https://<machine>.<tailnet>.ts.net URL
```

`--bg` persists the funnel, and the Tailscale service auto-starts on boot,
so the URL survives restarts. Pair the server with a Task Scheduler
"At log on" task (`run_server.cmd`) — see
[tailscale_funnel.md](tailscale_funnel.md) step 4 — and the whole site comes
back automatically after a reboot with no manual steps. The site is
reachable while your PC is powered on.

## Notes

- `gee_key.json` (the Earth Engine service-account key) stays local and
  gitignored — never commit it.
- GEE quota is per-project regardless of host; keep the job concurrency cap
  (default 2) and the per-IP rate limits enabled.

> **Want an always-on cloud VM instead?** The repo still ships the pieces —
> `vm_setup.sh`, `Dockerfile`, `krishidrishti.service`, `Caddyfile` — see
> their header comments. This README documents the Tailscale setup in use.
