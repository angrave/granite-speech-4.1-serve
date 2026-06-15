# Linux systemd Service — granite-speech-4.1

Runs the granite-speech docker compose stack automatically at boot using a
systemd **system service**. Container restarts are handled by docker compose's
own `restart: unless-stopped` policy; systemd's job is to start the stack at
boot and stop it cleanly on shutdown.

## Prerequisites

1. Docker installed and the daemon running (`systemctl status docker`).
2. `.env` exists at the repo root — copy `.env.example` and fill in
   `GRANITE_API_KEY` and `LLAMA_API_KEY`.
3. *(GPU only)* `nvidia-container-toolkit` installed for GPU passthrough.

## Setup

`--mode` is required. Choose `ghcr` (recommended) to pull pre-built images, or
`local` to build from the local Dockerfile.

```bash
# Recommended — pull pre-built images from ghcr.io
sudo bash service/linux-systemd/install.sh --mode ghcr

# Alternative — build locally from source
sudo bash service/linux-systemd/install.sh --mode local
```

### Log destination (optional)

By default, logs go to the systemd journal. To write to a file instead:

```bash
sudo bash service/linux-systemd/install.sh --mode ghcr --log file
```

## Daily commands

| Action | Command |
|--------|---------|
| Install / reinstall | `sudo bash service/linux-systemd/install.sh --mode ghcr` |
| Stop (keep installed) | `sudo bash service/linux-systemd/stop.sh` |
| Start / restart | `sudo bash service/linux-systemd/start.sh` |
| Remove from boot | `sudo bash service/linux-systemd/uninstall.sh` |
| Check status | `systemctl status granite-speech` |
| Watch logs (journal) | `journalctl -u granite-speech -f` |
| Watch logs (file) | `tail -f service/linux-systemd/service.log` |
| Watch container logs | `docker compose logs -f` (from repo root) |

## How it works

- **`Type=oneshot RemainAfterExit=yes`** — systemd runs the start script once;
  the script exits after `docker compose up -d` succeeds. systemd marks the unit
  active and keeps it that way until explicitly stopped.
- **`Requires=docker.service`** — ensures the Docker daemon is running before the
  stack starts.
- **Container restarts** — handled by docker compose's own `restart: unless-stopped`
  policy, not by systemd. Containers survive docker daemon restarts automatically.
- **Clean shutdown** — `ExecStop` runs `docker compose down` to stop all containers
  gracefully before the system powers off.
- **GPU detection** — done at runtime inside `start_ghcr.sh` / `start_local_docker.sh`.
  No extra configuration needed here.

## Files

| File | Purpose |
|------|---------|
| `granite-speech.service` | systemd unit template (placeholders filled in by `install.sh`) |
| `install.sh` | Expand template, install to `/etc/systemd/system/`, enable and start |
| `uninstall.sh` | Stop, disable, and remove unit file |
| `start.sh` | Start or restart without reinstalling |
| `stop.sh` | Stop without uninstalling |
| `service.log` | stdout + stderr from the service, only when `--log file` is used |

---

## Running as a user-level service instead (no root after initial setup)

The setup above installs a *system* service (runs at boot, managed by root).
If you prefer a *user-level* service — closer to the macOS LaunchAgent model,
no root needed day-to-day — follow these steps instead:

1. **Change the unit destination** in `install.sh`: replace
   ```
   UNIT_FILE="/etc/systemd/system/${UNIT_NAME}.service"
   ```
   with:
   ```
   UNIT_FILE="$HOME/.config/systemd/user/${UNIT_NAME}.service"
   mkdir -p "$(dirname "$UNIT_FILE")"
   ```

2. **Remove the `User=` line** from the unit template — a user service already
   runs as you.

3. **Replace every `systemctl` call** with `systemctl --user` (in `install.sh`,
   `start.sh`, `stop.sh`, and `uninstall.sh`).

4. **Remove the `[[ $EUID -eq 0 ]]` root check** from all scripts (and the
   `exec sudo` escalation lines) — user services don't need root.

5. **Enable lingering** so the service survives logout (requires root once):
   ```bash
   sudo loginctl enable-linger $USER
   ```
   Without this, systemd stops your user session (and all user services) when
   you log out.

6. **Set `XDG_RUNTIME_DIR`** if you get `Failed to connect to bus` errors on
   headless servers:
   ```bash
   export XDG_RUNTIME_DIR=/run/user/$(id -u)
   ```
   Add this to your shell profile or the service's `Environment=` line.
