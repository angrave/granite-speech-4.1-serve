Windows Deployment — granite-speech-4.1
========================================

This directory documents options for running the granite-speech docker compose
stack as a persistent service on Windows. No scripts or configs are provided —
the right approach depends on your environment, so choose the option that fits.

Prerequisites (all options)
----------------------------
- Docker Desktop for Windows (with WSL2 backend recommended)
- .env file in the repo root with GRANITE_API_KEY, LLAMA_API_KEY, and HF_TOKEN
- For NVIDIA GPU: install nvidia-container-toolkit and enable GPU in Docker Desktop

The containers themselves use "restart: unless-stopped", so once the stack is
started, Docker handles individual container restarts. The service layer only
needs to ensure the stack starts at boot and stops cleanly on shutdown.


Option A — WSL2 + Windows Task Scheduler (simplest)
-----------------------------------------------------
Best for: desktop/workstation machines where a user is always logged in.
Pros: zero extra dependencies; reuses the existing bash start scripts unchanged.
Cons: WSL2 is tied to the Windows session — does not run before login on a
      headless server.

How it works:
  A single Task Scheduler entry at system startup calls wsl.exe to invoke the
  existing start_ghcr.sh (or start_local_docker.sh) inside WSL2. Docker Desktop
  exposes the Docker engine to WSL2, so docker compose works natively inside the
  WSL2 shell.

Steps:
  1. Ensure WSL2 is installed with a Ubuntu distribution (22.04+ recommended).
  2. Clone the repo inside WSL2 (e.g. ~/granite-speech-4.1-serve) and create .env.
  3. Run the chosen start script manually once to verify it works:
       ./start_ghcr.sh
  4. Open Task Scheduler > Create Task:
       General:    "Run whether user is logged on or not", highest privileges
       Trigger:    At system startup
       Action:     Program: wsl.exe
                   Arguments: -d Ubuntu -- bash /home/<user>/granite-speech-4.1-serve/start_ghcr.sh
  5. To stop: open Task Scheduler and disable the task, then run from WSL2:
       docker compose down


Option B — WSL2 + systemd (Linux-native inside WSL2)
------------------------------------------------------
Best for: developers who are comfortable with Linux and want the same service
          management experience as the linux-systemd setup (see service/linux-systemd/).
Pros: identical to the Linux setup; uses the existing install.sh scripts as-is.
Cons: requires systemd enabled in WSL2 (WSL2 version 0.67.6+ and Ubuntu 22.04+);
      still subject to the WSL2 session-lifecycle limitation on headless machines.

How it works:
  Enable systemd in WSL2 by adding to /etc/wsl.conf inside your WSL2 distro:
    [boot]
    systemd=true
  Then restart WSL2 (wsl --shutdown from PowerShell, then reopen).
  After that, the service/linux-systemd/ scripts work without modification:
    sudo bash service/linux-systemd/install.sh --mode ghcr

To have WSL2 start automatically on Windows boot (so systemd services run),
add a Task Scheduler entry at startup that runs:
    wsl.exe -d Ubuntu -- echo "WSL2 started"
This wakes WSL2 early; your systemd service then starts the stack.


Option C — NSSM (Non-Sucking Service Manager) + PowerShell
-----------------------------------------------------------
Best for: headless Windows servers where the machine boots unattended and no
          user session is expected.
Pros: proper Windows Service — appears in services.msc, starts before login,
      restarts on failure, logs to Windows Event Log.
Cons: requires NSSM (third-party, ~300 KB exe); GPU detection logic from the
      bash start scripts must be re-implemented in PowerShell (~10 lines).

How it works:
  NSSM wraps a PowerShell script as a Windows Service registered with the SCM.
  The PowerShell script replicates the logic of start_ghcr.sh or
  start_local_docker.sh: detect nvidia-smi, set the appropriate compose file
  flags, then call docker compose up -d. NSSM handles restart-on-failure and
  lifecycle management.

NSSM is available at https://nssm.cc (no installer; single exe).

Rough steps:
  1. Download nssm.exe and place it somewhere on your PATH.
  2. Write a PowerShell start script that calls docker compose with the
     appropriate -f flags (see start_ghcr.sh for the logic to translate).
  3. Register it as a service:
       nssm install granite-speech powershell.exe -File "C:\path\to\start.ps1"
       nssm set granite-speech AppDirectory "C:\path\to\repo"
       nssm set granite-speech Start SERVICE_AUTO_START
       nssm start granite-speech
  4. To remove:
       nssm stop granite-speech
       nssm remove granite-speech confirm


Option D — WinSW (Windows Service Wrapper)
------------------------------------------
Best for: same use case as NSSM but preferred if you want an actively maintained
          alternative (NSSM's last release was 2017).
Pros: XML-config-driven, no separate installer, actively maintained.
Cons: same PowerShell translation requirement as NSSM; one XML config file per
      service.

WinSW is available at https://github.com/winsw/winsw.
Configuration and usage are documented in the WinSW repository.


Option E — Windows Task Scheduler only (no WSL2)
--------------------------------------------------
Best for: simple setups where Docker Desktop is already running and you just
          need the compose stack to start at login.
Pros: zero extra dependencies; built into Windows.
Cons: no restart-on-failure semantics; harder to script; tied to user login.

Steps:
  1. Create a .bat file in the repo root:
       docker compose -f docker-compose.yml -f docker-compose.ghcr.yml up -d
     (add -f docker-compose.gpu.yml if using an NVIDIA GPU)
  2. Add a Task Scheduler entry (At log on / current user or all users)
     pointing to the .bat file with "Start in" set to the repo root.


GPU notes (all options)
------------------------
- NVIDIA GPU passthrough through Docker Desktop on WSL2 requires:
    nvidia-container-toolkit installed inside WSL2
    NVIDIA drivers on the Windows host (no separate Linux driver needed in WSL2)
    Docker Desktop "Use GPU" enabled in Settings > Resources > Advanced
- The start scripts (start_ghcr.sh, start_local_docker.sh) auto-detect nvidia-smi
  and select the appropriate image tag (:cuda) and compose overlay automatically.
  If re-implementing in PowerShell, replicate this check:
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { ... }
