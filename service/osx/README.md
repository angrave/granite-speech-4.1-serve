# macOS LaunchAgent — granite-speech-4.1

Runs `start_apple_dockerless.sh` automatically at login using macOS's native
`launchd`. The service restarts itself if it crashes (`KeepAlive`).

## Prerequisites

1. Run `scripts/start_apple_dockerless.sh` **manually at least once** from the repo root.
   First-run downloads several GB of models and may build llama.cpp from source
   (~10 min). The service is only suitable for subsequent auto-starts.
2. Ensure `.env` exists in the repo root with valid `GRANITE_API_KEY` and
   `LLAMA_API_KEY` values.

## Setup

```bash
cd service/osx
bash install.sh
```

This fills in the paths in the plist template, copies it to
`~/Library/LaunchAgents/`, and loads it. The servers start immediately and
will start again on every login.

## Daily commands

| Action | Command |
|--------|---------|
| Install / reinstall | `bash service/osx/install.sh` |
| Stop (keep installed) | `bash service/osx/stop.sh` |
| Start / restart now | `bash service/osx/start.sh` |
| Remove from login | `bash service/osx/uninstall.sh` |
| Watch logs | `tail -f service/osx/service.log` |
| Check status | `launchctl list com.granite-speech.serve` |
| Watch server logs | `tail -f runtime/logs/*.log` |

## How it works

- **LaunchAgent** (not LaunchDaemon) — runs in your user login session so
  MPS/Metal, Homebrew (`/opt/homebrew/bin`), and your home directory are all
  accessible.
- **`KeepAlive true`** — launchd restarts the script if it exits.
- **`ThrottleInterval 30`** — prevents a tight restart loop if startup fails
  (e.g., network not yet available after boot).
- **`RunAtLoad true`** — starts the service immediately when the plist is
  loaded, not just at the next login.

## Files

| File | Purpose |
|------|---------|
| `com.granite-speech.serve.plist` | Plist template (`INSTALL_DIR` / `HOME_DIR` filled in by `install.sh`) |
| `install.sh` | Expand template, install to `~/Library/LaunchAgents/`, load |
| `uninstall.sh` | Unload and remove from `~/Library/LaunchAgents/` |
| `start.sh` | Start or restart without reinstalling |
| `stop.sh` | Stop without uninstalling |
| `service.log` | stdout + stderr from the service (created at runtime) |
