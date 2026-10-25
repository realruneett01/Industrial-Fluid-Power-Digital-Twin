# Deployment and verification guide

Confidential. See `LICENSE`.

## What you need on the target machine

| Path | Needs | Notes |
| --- | --- | --- |
| USB drive, Linux / macOS / Raspberry Pi (ARM64) | Python 3.12+ | `./start-hydro.sh` |
| USB drive, Windows 10/11 | Python 3.12+ | `Start-Hydro.bat` |
| Docker | Docker Engine | no host Python needed |
| systemd service | Linux with systemd, Python 3.12+ | `deploy/hydro.service` |

Sensors are read over **OPC-UA** only. Set the endpoint and credentials in the environment (never on the drive):

```bash
export HYDRO_OPCUA_ENDPOINTS=opc.tcp://plc.local:4840/hydro/
export HYDRO_OPCUA_USER=...      # Windows: set HYDRO_OPCUA_USER=...
export HYDRO_OPCUA_PASSWORD=...
```

Encrypted sessions (the default) pin the server's certificate: copy it to `site/certs/server_cert.der` before the first run, otherwise the gate fails with a `server-certificate` error. `--insecure` skips encryption and belongs only on an isolated lab network.

## Build the drive (on the engineering machine)

```bash
pip install -e ".[train]"
python -m hydro ingest                                   # downloads UCI 447 (needs internet)
python -m hydro bundle --out models/online               # trains and exports the four ONNX models
python scripts/build_portable.py --out /media/usb/hydro --models models/online \
    --platform linux-x86_64 --platform linux-aarch64 --platform windows-x64 \
    --python-version 3.12 --python-version 3.13
```

`build_portable.py` copies the program, the models and offline wheels for each platform and Python version (an offline install needs an exact match, so list every Python version your target machines run), then writes `MANIFEST.sha256.json`. Set `HYDRO_MANIFEST_KEY` before building (and on the target) to sign it. Do not run Python from the drive after the manifest is built without `PYTHONDONTWRITEBYTECODE=1` (the launchers set it).

Models are not in git (`/models/` and `*.onnx` are ignored): they exist only on machines that built them.

## Run on the target

```bash
./start-hydro.sh --check-only      # gate only; prints every check, exit 0 = may deploy, 2 = blocked
./start-hydro.sh                   # gate, then start
```

Windows: `Start-Hydro.bat --check-only`, then `Start-Hydro.bat`.

Exit codes: `0` ok, `2` gate failed (nothing was started), `3` no usable Python, `1` other error.

Output files, all under `site/` (override with `HYDRO_SITE_DIR`): `site.json` (auto-configuration), `calibration.json` (baseline), `deploy_report.json` (full evidence), `heartbeat.json` (liveness), `alerts.jsonl` (alerts when no CMMS is configured).

## Commands that check what is needed

Run these on each platform. A healthy result is stated after each command.

### Any platform (the same checks the launcher runs)

```bash
python --version                         # 3.12 or newer
pip install -e .                         # runtime only
hydro doctor                             # every line PASS (exit 0)
hydro manifest verify                    # ok: True (needs a built manifest)
hydro deploy --mode live --endpoint opc.tcp://HOST:4840/hydro/   # VERDICT: PASS or WARN
hydro health                             # exit 0 while the service runs
pytest -q                                # full test suite
```

### Windows (PowerShell)

```powershell
py -3.12 --version
py -3.12 -m venv $env:LOCALAPPDATA\hydro\venv-test
& $env:LOCALAPPDATA\hydro\venv-test\Scripts\pip install -e .
& $env:LOCALAPPDATA\hydro\venv-test\Scripts\hydro doctor
.\Start-Hydro.bat --check-only ; echo "exit code: $LASTEXITCODE"
```

Expected without sensors and without a model bundle: exit code `2` (the gate fails closed). With a bundle and a dataset on the drive: exit code `0` and a WARN that it is a replay.

### ARM (Raspberry Pi 4/5 with 64-bit OS, or any aarch64 Linux)

```bash
uname -m                                  # aarch64
python3 --version                         # 3.12+ (Raspberry Pi OS Bookworm ships 3.11: install 3.12 first)
./start-hydro.sh --check-only
cat site/deploy_report.json | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['profile'])"
```

Watch the `latency-sla` line: a Pi is much slower than a workstation, and the gate blocks deployment when p95 exceeds 150 ms. 32-bit ARM is not supported (no wheels for the dependencies).

### Docker

```bash
docker build -t hydro:0.2 .
docker run --rm hydro:0.2 deploy ; echo "exit code: $?"      # 2 with no sensors (fails closed)
docker run -d --name hydro \
  -e HYDRO_OPCUA_USER -e HYDRO_OPCUA_PASSWORD -e HYDRO_OPCUA_ENDPOINTS \
  -v hydro-site:/var/lib/hydro/site hydro:0.2 deploy --start
docker inspect -f '{{.State.Health.Status}}' hydro            # healthy
docker exec hydro hydro health
docker logs hydro
```

The image builds from a checkout that contains `models/online`. It runs as an unprivileged user and contains no training libraries.

### systemd

```bash
systemd-analyze verify deploy/hydro.service        # static check (works before installing)
sudo systemctl daemon-reload && sudo systemctl enable --now hydro
systemctl status hydro
journalctl -u hydro -f
sudo systemctl show hydro -p ExecMainStatus
systemd-analyze security hydro                      # hardening score
```

## What has actually been verified

| Target | Verified how | Result |
| --- | --- | --- |
| Linux x86_64, Python 3.13 | full test suite and an end-to-end gate/service run against a synthetic dataset and a live local OPC-UA stream | passed |
| Docker (linux/amd64) | image built, gate failed closed (exit 2) with no sensors, replay service started and reported `healthy`, ran as a non-root user | passed |
| systemd | `systemd-analyze verify` (static) | passed except the expected "binary not installed" notice; **not** run under a live systemd |
| Windows | not run (no Windows host available); launcher and code paths written to be portable; CI job `runtime-only (windows-latest)` runs `Start-Hydro.bat --check-only` | **unverified until CI runs** |
| ARM64 | not run (no ARM host or emulation available); CI job on `ubuntu-24.04-arm` | **unverified until CI runs** |

Run the CI workflow (`.github/workflows/ci.yml`) once the repository is private and hosted to close the two unverified rows. Treat Windows and ARM as unproven until it is green.

## Known gaps

- The real UCI dataset was not available in the build sandbox, so the `--heavy-check` real-data tests and the accuracy figures in the README were not re-run there.
- The latency SLA gate uses synthetic cycles; real cycles take the same code path but values differ.
- Plausibility limits for calibration are wide sensor spans, not data-derived bands.
- Live capture needs about one minute per cycle, so a live gate takes about two minutes at the default.
