# Deployment

The public site runs from PC, exposed with a free **Tailscale
Funnel** — a stable `https://<machine>.<tailnet>.ts.net` URL. 
**[tailscale_funnel.md](tailscale_funnel.md)**.

## PC + Tailscale Funnel

```powershell
# 1. Server — serves the built frontend + API on :8000
start_servers.cmd            # or auto-start via Task Scheduler (see below)
tailscale funnel --bg 8000
tailscale funnel status      # prints the public https://<machine>.<tailnet>.ts.net URL
```

`--bg` persists the funnel, and the Tailscale service auto-starts on boot,
so the URL survives restarts. The server is paired with a Task Scheduler
"At log on" task (`run_server.cmd`) — see
[tailscale_funnel.md](tailscale_funnel.md) step 4 — and the whole site comes
back automatically after a reboot with no manual steps. The site is
reachable while the PC is powered on.
