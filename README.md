# Hydraulic Condition Monitoring Digital Twin (`hydro`)

> **Proprietary and confidential. All rights reserved.** This is not open-source software. No one may use, copy, modify or share it without a signed written agreement with the Owner. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

<p align="center"><img src="docs/assets/test_stand_3d.svg" alt="Isometric animated view of the ZeMA hydraulic test stand with live health index" width="100%"></p>

A read-only condition monitoring and diagnostic digital twin for the ZeMA hydraulic test stand (UCI 447, 2,205 cycles × 60 s), with a self-checking plug-and-play deployment layer.

**Status: research prototype.** The models are trained and evaluated on a public laboratory dataset. The OPC-UA service has been exercised against a replay server and unit/integration tests, not against a live production plant. Diagnostic output is advisory and must not be the only safeguard for equipment or people.

## The problem and how we solved it (37 s)

<p align="center"><a href="docs/assets/explainer.mp4"><img src="docs/assets/explainer.gif" alt="37-second explainer film of hydraulic condition monitoring problem, solution, and verified results" width="100%"></a></p>

Click for the full-quality MP4.

| Problem | What we built | Verified result |
| --- | --- | --- |
| Hydraulic wear in cooler, valve, pump, and accumulator is invisible until functional failure. | Multi-target CatBoost models combining statistical features with grey-box physics indicators (Walther viscosity, hydraulic capacitance, valve edge dynamics). | Under blocked folds, LDA is the strongest baseline on several targets (cooler 0.9823, valve 0.9977, pump 0.9915, accumulator 0.8019); hybrid CatBoost reaches 0.8792 cooler, 0.9764 valve, 0.9089 pump; CatBoost passes ONNX export parity (rtol 1e-5, atol 1e-6). |
| 17 sensor channels across three acquisition rates (100 Hz, 10 Hz, 1 Hz) generate too much raw data to monitor by eye. | Single-cycle feature extraction and ONNX inference served over an encrypted, read-only OPC-UA interface (validated against a replay server). | CPU inference latency of 15.29 ms (p95) on the author's workstation (SLA < 150 ms; the deploy gate re-measures this on every target machine); sensor nodes are read-only and a runtime self-check proves writes are refused. |
| Unplanned downtime and lack of automated maintenance ticket triggering upon fault detection. | Online supervisory advisor calculating per-cycle Health Index (HI) with automated work order dispatch to ERPNext CMMS. | Degraded condition detected at HI ≈ 0.26 and recovery at HI ≈ 0.51 in an author-run replay of the dataset (needs the dataset to reproduce); automated test suite included (`pytest`). |

## What It Does

- Diagnoses four component fault modes using multi-rate sensor telemetry:
  1. **Cooler condition**: full efficiency (100 %), reduced efficiency (20 %), or close to total failure (3 %).
  2. **Valve condition**: optimal switching (100 %), small lag (90 %), severe lag (80 %), or close to total failure (73 %).
  3. **Internal pump leakage**: no leakage (0), weak leakage (1), or severe leakage (2).
  4. **Hydraulic accumulator**: optimal pressure (130 bar), slightly reduced (115 bar), severely reduced (100 bar), or severely compromised (90 bar).
- Extracts statistical time-series features alongside grey-box hydraulic physics:
  - Walther oil viscosity equation (ASTM D341) for temperature-dependent fluid behavior.
  - Lumped hydraulic capacitance $C_{\mathrm{eff}}$ during pressure build-up.
  - Directional valve switching edge dynamics and inflection delays.
  - Causal heat exchanger thermal effectiveness over multi-cycle trailing windows.
- Evaluates multi-target classification under leakage-free blocked 5-fold cross-validation and temperature-shift distribution drift.
- Runs as a read-only OPC-UA client/service with per-cycle Health Index (HI) estimation and maintenance work-order dispatch (ERPNext CMMS, or a local `alerts.jsonl` when no CMMS is configured).
- Ships a plug-and-play deployment gate: environment check, sensor discovery, calibration/acceptance, self-check and integrity verification before anything starts (see [Plug-and-play deployment](#plug-and-play-deployment)).

## Hydraulic Circuit

<p align="center"><img src="docs/assets/hydraulic_circuit.svg" alt="ZeMA hydraulic test stand circuit diagram" width="100%"></p>

## Architecture

<p align="center"><img src="docs/assets/system_pipeline.svg" alt="End-to-end system architecture and data pipeline" width="100%"></p>

The processing pipeline is organized into four distinct layers:
1. **Data Ingestion & Interim Store**: Validates raw sensor telemetry, verifies SHA-256 digest (TOFU), and stores partitioned interim Parquet files.
2. **Feature Extraction**: Computes statistical features (moments, energy, percentiles) and grey-box physics indicators (Walther viscosity, $C_{\mathrm{eff}}$, valve edge delays, thermal effectiveness).
3. **Model Pipelines & ONNX Export**: Trains classical classifiers (LDA, Random Forest, CatBoost) and a multi-rate 1D-CNN, with automated export and numerical parity validation against ONNX Runtime.
4. **Online Supervisory Advisor**: Replays cycles over an OPC-UA server with read-only node access levels, estimates component health indices, and dispatches maintenance work orders to ERPNext CMMS.

## Data

<p align="center"><img src="docs/assets/condition_timeline.svg" alt="Condition monitoring timeline across 2,205 cycles" width="100%"></p>

The dataset comprises 2,205 operating cycles of 60 seconds each, acquired on the ZeMA hydraulic test rig:
- **100 Hz Channels** (6,000 samples/cycle): `PS1`–`PS6` (hydraulic pressures), `EPS1` (motor electrical power).
- **10 Hz Channels** (600 samples/cycle): `FS1` (primary flow rate), `FS2` (secondary flow rate).
- **1 Hz Channels** (60 samples/cycle): `TS1`–`TS4` (fluid temperatures), `VS1` (vibration).
- **Virtual Channels** (`CE`, `CP`, `SE`): Excluded by default (`exclude_virtual=True`) to avoid target leakage.

### Animation Regeneration

To regenerate the animated SVG assets:

```bash
python scripts/make_readme_animation.py
```

To regenerate the explainer video, GIF, and WebP (scene images in `docs/assets/explainer/`, generated with Gemini image models in the style of a photoreal test-stand reference):

```bash
python scripts/make_explainer_video.py
```

## Recorded Results

These figures were produced by the author's runs on the full dataset and are copied from `reports/`. They are not regenerated by the fast test suite; reproducing them needs the dataset (`python -m hydro ingest`) and, for the CNN/CatBoost comparison, a GPU-capable machine.

| Metric / Characteristic | Value | Specification / Standard |
| --- | --- | --- |
| Operating cycles | 2,205 | Full ZeMA UCI 447 run |
| Feature extraction + inference latency (CPU p50, author's workstation) | 13.89 ms | SLA < 150 ms |
| Feature extraction + inference latency (CPU p95) | 15.29 ms | SLA < 150 ms |
| Feature extraction + inference latency (CPU p99) | 15.97 ms | SLA < 150 ms |
| CatBoost ONNX export parity | Passes allclose | rtol 1e-5, atol 1e-6 (max rel error 1.82e-07) |
| 1D-CNN ONNX export parity | Validated (looser than plan target 1e-5) | rtol 1e-4, atol 1e-5 (max rel error ~3.3e-4) |
| Automated test suite | `pytest` | Fast suite uses synthetic fixtures; `--heavy-check` adds the real-data suite and needs the dataset |
| Online replay (cycles 2100–2105: valve 90, pump 2, acc 90), author-run | HI ≈ 0.26 | Degraded condition detected |
| Online replay (valve 100 % restoration), author-run | HI ≈ 0.51 | Partial recovery detected |

### Baseline Evaluation (Blocked 5-Fold)

From `reports/baseline_results.md`:

| pipeline | target | macro_f1 | macro_f1_all_labels | balanced_accuracy | fit_seconds |
| --- | --- | --- | --- | --- | --- |
| lda | cooler | 0.9823 | 0.9823 | 0.9823 | 0.7975 |
| lda | valve | 0.9977 | 0.9977 | 0.9977 | 0.7277 |
| lda | pump | 0.9915 | 0.9915 | 0.9917 | 0.7304 |
| lda | accumulator | 0.8019 | 0.8019 | 0.7981 | 0.7233 |
| rf | cooler | 0.7880 | 0.7880 | 0.7956 | 2.5896 |
| rf | valve | 0.9324 | 0.9324 | 0.9276 | 2.3430 |
| rf | pump | 0.8955 | 0.8955 | 0.8977 | 2.2938 |
| rf | accumulator | 0.1426 | 0.1426 | 0.1625 | 2.5306 |

### Four-Way Comparison (Blocked 5-Fold)

Training ran on an RTX 5070 Ti GPU (CatBoost task_type GPU, CNN on CUDA) with CPU-identical CatBoost hyperparameters.

From `reports/benchmark_results.md`:

| pipeline | target | macro_f1 | macro_f1_all_labels | balanced_accuracy | fit_seconds |
| --- | --- | --- | --- | --- | --- |
| statistical | cooler | 0.8487 | 0.8487 | 0.8507 | 21.3618 |
| statistical | valve | 0.9700 | 0.9700 | 0.9707 | 22.8830 |
| statistical | pump | 0.8854 | 0.8854 | 0.8895 | 20.8299 |
| statistical | accumulator | 0.3149 | 0.3149 | 0.3258 | 23.1180 |
| physics | cooler | 0.8185 | 0.8185 | 0.8293 | 16.6824 |
| physics | valve | 0.8462 | 0.8462 | 0.8815 | 16.9406 |
| physics | pump | 0.8112 | 0.8112 | 0.8002 | 17.0056 |
| physics | accumulator | 0.4221 | 0.4221 | 0.4373 | 18.1276 |
| hybrid | cooler | 0.8792 | 0.8792 | 0.8821 | 22.1368 |
| hybrid | valve | 0.9764 | 0.9764 | 0.9771 | 23.2573 |
| hybrid | pump | 0.9089 | 0.9089 | 0.9126 | 21.9153 |
| hybrid | accumulator | 0.3253 | 0.3253 | 0.3215 | 23.9561 |
| cnn_1d | cooler | 0.8756 | 0.8756 | 0.8785 | 9.1818 |
| cnn_1d | valve | 0.9972 | 0.9972 | 0.9963 | 9.1818 |
| cnn_1d | pump | 0.8856 | 0.8856 | 0.8858 | 9.1818 |
| cnn_1d | accumulator | 0.2127 | 0.2127 | 0.2103 | 9.1818 |

### Temperature-Shift Generalization

From `reports/benchmark_results.md`:

| pipeline | target | f1_in | f1_shift | f1_shift_all_labels | delta_f1 |
| --- | --- | --- | --- | --- | --- |
| statistical | cooler | 0.6579 | 0.2640 | 0.1760 | -0.3939 |
| statistical | valve | 0.9991 | 0.7853 | 0.7853 | -0.2138 |
| statistical | pump | 1.0000 | 0.6110 | 0.6110 | -0.3890 |
| statistical | accumulator | 0.2279 | 0.1303 | 0.1303 | -0.0976 |
| physics | cooler | 0.6533 | 0.2717 | 0.1811 | -0.3816 |
| physics | valve | 0.9725 | 0.7228 | 0.7228 | -0.2497 |
| physics | pump | 0.7937 | 0.4607 | 0.4607 | -0.3330 |
| physics | accumulator | 0.3970 | 0.2815 | 0.2815 | -0.1155 |
| hybrid | cooler | 0.6579 | 0.2624 | 0.1749 | -0.3955 |
| hybrid | valve | 1.0000 | 0.7228 | 0.7228 | -0.2772 |
| hybrid | pump | 1.0000 | 0.6110 | 0.6110 | -0.3890 |
| hybrid | accumulator | 0.3436 | 0.2741 | 0.2741 | -0.0695 |
| cnn_1d | cooler | 0.6567 | 0.2670 | 0.1780 | -0.3897 |
| cnn_1d | valve | 0.9871 | 0.5005 | 0.5005 | -0.4865 |
| cnn_1d | pump | 0.9644 | 0.5101 | 0.5101 | -0.4543 |
| cnn_1d | accumulator | 0.1463 | 0.0814 | 0.0814 | -0.0649 |

## Plug-and-play deployment

Copy the program folder to a USB drive, plug it into the target computer and start it. The launcher finds Python, builds a private environment (offline from the bundled wheels when present), and then runs the **deployment gate** before anything is started:

| Step | What it does | Fails the gate when |
| --- | --- | --- |
| Machine check (`doctor`) | Python version, required libraries, writable and free space, system clock | a library is missing or broken, the drive is read-only or full, the clock is before 2024 |
| Integrity | Verifies every file against the SHA-256 manifest (optionally HMAC-signed) | any file is missing, changed, or the signature does not match |
| Discovery | Probes OPC-UA endpoints (`HYDRO_OPCUA_ENDPOINTS`, defaults) and looks for datasets on the drive | nothing to read from (no sensor stream and no dataset) |
| Calibration / acceptance | Captures complete cycles and checks every channel: sample count, NaN/inf, physical range, flat (dead) signal, noise; stores a baseline in `site/calibration.json` | a channel is missing, mis-scaled, out of range or mostly invalid (`--strict` also fails on warnings) |
| Self-check | Loads the model bundle, validates the probabilities, **measures latency on this machine against the 150 ms SLA**, and proves that a client write to a sensor node is refused | anything is off, including a write succeeding |
| Auto-configuration | Chooses ONNX threads, OPC-UA publish rate and calibration length from CPU/RAM; writes `site/site.json` | - |

Nothing starts unless every step is PASS or WARN; a failed gate exits with code 2 and the full evidence is written to `site/deploy_report.json`.

```bash
./start-hydro.sh                 # Linux / macOS / Raspberry Pi: gate, then start the service
./start-hydro.sh --check-only    # gate only
Start-Hydro.bat                  # Windows (double-click)
```

Important limits, stated plainly:

- **Calibration here is acceptance and baselining, not model re-tuning.** It tells you whether the attached sensors deliver usable, plausible cycles and records a baseline to measure drift against. The trained models are not adjusted. Plausibility limits are wide sensor-span limits (e.g. 0-700 bar), not data-derived bands, so they catch wiring and scaling faults but not subtle miscalibration.
- **Live sensors are read through OPC-UA only.** The drive does not speak Modbus, CAN or raw serial; serial ports are listed for information. A gateway that exposes the ZeMA-style block frame (see `src/hydro/online/replay_server.py`) is required.
- **With no live endpoint, the gate falls back to replaying a dataset on the drive and says so (WARN, "REPLAY, not live").** That is a demo mode.
- The host needs Python 3.12 or newer (or use Docker). Copy-protection is not provided; the manifest is tamper *evidence*.
- Credentials for live OPC-UA come from `HYDRO_OPCUA_USER` / `HYDRO_OPCUA_PASSWORD`; they are never written to the drive. The loopback replay mode generates a random password in `site/credentials.json` (file mode 0600 on POSIX; FAT/exFAT drives cannot enforce permissions, so keep `site/` on the host disk with `HYDRO_SITE_DIR` there).

Build the drive image with `python scripts/build_portable.py --out <drive> --models models/online`. Docker, systemd, Windows and ARM instructions, including the commands to verify each, are in [docs/deploy/DEPLOYMENT.md](docs/deploy/DEPLOYMENT.md).

## Quick Start

### 1. Environment Setup

Requires Python 3.12 or newer (tested on 3.12 and 3.13). `pip install -e .` installs only the lightweight runtime; `.[train]` adds the training stack (scikit-learn, CatBoost, PyTorch, ONNX, PyArrow) and `.[dev]` adds that plus the test tools:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .[dev]
```

### 2. Ingest Dataset

Download, verify SHA-256 (TOFU), and partition raw data to Parquet:

```bash
python -m hydro ingest
# Or use the wrapper:
python scripts/run_ingest.py
```

### 3. Run Test Suite

Run unit and integration tests (synthetic fast fixtures finish in seconds):

```bash
pytest -q
```

### 4. Run Benchmark Evaluations

Run blocked 5-fold cross-validation baseline:

```bash
python -m hydro eval --config configs/data.yaml --split blocked --mode baseline-only
```

Run temperature-shift generalization benchmark:

```bash
python -m hydro eval --config configs/data.yaml --split temp-shift --mode full-comparison
# Or use the wrapper:
python scripts/run_benchmark.py
```

### 5. Profile Inference Latency

Measure feature extraction and ONNX runtime inference latency against SLA gates:

```bash
python scripts/profile_latency.py --cycles 10
```

### 6. Online Replay + Advisor

Train and export the online model bundle (last 205 cycles held out for replay), then replay held-out cycles over OPC-UA and run the read-only advisor on them:

```bash
python -m hydro bundle --out models/online
export HYDRO_OPCUA_USER=<user> HYDRO_OPCUA_PASSWORD=<password>   # local test credentials
python -m hydro.online.replay_server --start 2100 --cycles 6 --speed 5 &
python -m hydro.online.advisor_client --models-dir models/online --duration-sec 75
```

Application certificates are created automatically in `certs/` on first run (gitignored, valid 365 days). Both sides must share the `certs/` directory (same machine) so the advisor client can pin the server certificate.

For localhost testing without transport encryption, use the `--insecure` option:

```bash
python -m hydro.online.replay_server --insecure --start 2100 --cycles 6 --speed 5 &
python -m hydro.online.advisor_client --insecure --models-dir models/online --duration-sec 75
```

Author-run result (2026-10-20, needs the dataset to reproduce): cycles 2100-2105 (valve 90 %, pump 2, accumulator 90) give HI about 0.26; a cycle with valve back at 100 % gives HI 0.51.

## Repository Layout

```
hydraulic-condition-twin/
├── configs/
│   ├── data.yaml             # Dataset paths, TOFU digest, split configurations
│   ├── model.yaml            # Hyperparameters and random seeds
│   └── oil_vg46.yaml         # HLP 46 oil physical properties
├── reports/
│   ├── baseline_results.md   # Blocked 5-fold baseline evaluation metrics
│   ├── benchmark_results.md  # Four-way comparison and temperature-shift report
│   └── latency.md            # Inference latency percentiles (p50, p95, p99)
├── scripts/
│   ├── profile_latency.py    # Latency benchmarking and SLA gating
│   ├── run_benchmark.py      # Entry point for full benchmark evaluation
│   └── run_ingest.py         # Entry point for raw dataset download & interim export
├── src/hydro/
│   ├── cli.py                # Command line interface (ingest, eval, replay, bundle)
│   ├── data/                 # Download, loader, and channel schema registry
│   ├── eval/                 # Splits, metrics, reporting, and benchmark pipelines
│   ├── features/             # Statistical, Walther viscosity, and physics features
│   ├── models/               # Classical pipelines (LDA, RF, CatBoost), 1D-CNN, ONNX export
│   ├── online/               # OPC-UA replay server, advisor client, and CMMS dispatcher
│   └── deploy/               # Plug-and-play gate: doctor, discovery, calibration, self-check, manifest, service
├── deploy/                   # systemd unit and environment template
├── docs/deploy/              # Deployment and verification guide, IP assessment
├── Dockerfile                # Runtime image (no training stack)
├── start-hydro.sh            # Linux / macOS / ARM launcher
├── Start-Hydro.bat           # Windows launcher
└── tests/                    # Unit, regression, security, fuzz and deployment tests
```

## Design Rules

- **Virtual Channels Excluded**: Calculated virtual channels (`CE`, `CP`, `SE`) are dropped by default (`exclude_virtual=True`) to prevent target leakage.
- **Blocked Cross-Validation**: Contiguous 5-fold time-ordered splits with zero temporal leakage across train and test partitions, plus fold class-coverage auditing.
- **Fold-Fitted Scalers and Baselines**: All transformations (e.g. `RobustScaler`, healthy valve inflection time reference $t_{\mathrm{baseline\_median}}$) are strictly fitted on training folds.
- **Causal Thermal Window**: Rolling multi-cycle thermal effectiveness features use a causal trailing window ($k - W + 1 \dots k$ with `min_periods=1`) to prevent future information leakage.
- **Read-Only OPC-UA**: All sensor nodes enforce `AccessLevel = UserAccessLevel = CurrentRead`. The server's custom user manager never grants the `Admin` role because `asyncua` Admin privileges bypass node `AccessLevel` restrictions.
- **Secure CMMS & OPC-UA Credentials**: Credentials and endpoints are configured strictly via environment variables (`ERPNEXT_URL`, `ERPNEXT_API_KEY`, `ERPNEXT_API_SECRET` for CMMS; `HYDRO_OPCUA_USER`, `HYDRO_OPCUA_PASSWORD` for OPC-UA). No secrets are stored in code or repository configuration files.

## Results

Detailed performance metrics and latency benchmarks are regenerated directly from data runs:
- Baseline evaluations (LDA, Random Forest): [reports/baseline_results.md](reports/baseline_results.md)
- Four-way comparison and temperature-shift generalization: [reports/benchmark_results.md](reports/benchmark_results.md)
- Single-cycle feature extraction and ONNX inference latency: [reports/latency.md](reports/latency.md)

## Limitations

- **Untested live CMMS**: ERPNext Issue doctype mapping is validated against local HTTP mocks and remains untested against a live production ERPNext server.
- **OPC-UA transport security**: OPC-UA uses Basic256Sha256 SignAndEncrypt with self-signed certificates; the server does not yet restrict client certificates to a trust list, so authentication relies on username/password; a plant deployment should use CA-issued certificates and a trust list.
- **Accumulator performance under blocked folds**: The hydraulic accumulator state is the weakest classification target for the tree and neural pipelines under blocked 5-fold cross-validation (LDA reaches 0.8019; Random Forest macro-F1 0.1426, Statistical CatBoost 0.3149, 1D-CNN 0.2127, Hybrid 0.3253, Physics 0.4221) due to severe fold imbalance and missing classes in test partitions.
- **Temperature-shift generalization hypothesis**: The hypothesis that grey-box physics features would resist performance degradation under thermal distribution drift was not confirmed by the data; all four pipelines experienced comparable F1 drops under temperature shift (e.g. cooler delta-F1 -0.3816 for physics vs -0.3939 for statistical).
- **CNN ONNX export parity**: While CatBoost passes allclose at rtol 1e-5 and atol 1e-6 (max relative error 1.82e-07), the 1D-CNN is validated at rtol 1e-4 and atol 1e-5 (max relative error about 3.3e-4 on near-zero outputs), looser than the plan target of 1e-5 (see `src/hydro/models/export.py`).
- **Laboratory test stand**: Sensor data originates from a controlled laboratory test bench (ZeMA) rather than an industrial field installation.
- **Identifiability of bulk modulus**: Effective fluid bulk modulus ($\beta_{\mathrm{eff}}$) is unidentifiable without geometric chamber volume measurements; only lumped capacitance $C_{\mathrm{eff}}$ is identified.
- **Artificially induced faults**: Faults were manually induced via valve settings and pre-charge changes rather than natural run-to-failure mechanical degradation.
- **Discrete fault states**: The dataset contains discrete fault state runs without continuous run-to-failure degradation trajectories.
- **Operating temperature band**: Oil temperatures remain within a 35–55 °C operating band, limiting extrapolation to extreme thermal regimes.
- **Single-cycle LMTD limitation**: True log mean temperature difference cannot be computed for a single cycle without secondary cooling circuit instrumentation.
- **Absence of solenoid timestamps**: Direct electrical switching signals are unavailable, necessitating pressure transient edge detection.
- **Deployed models are CatBoost, whose accumulator accuracy is weak**: the online bundle (`python -m hydro bundle`) trains CatBoost for all four targets. Under blocked folds CatBoost reaches macro-F1 0.33-0.42 on the accumulator versus 0.80 for LDA, so the Health Index inherits an unreliable accumulator term.
- **No field validation**: nothing in this repository has been validated on a production plant, a different machine type, or a different oil than the assumed ISO VG 46. Applying it elsewhere requires new data and retraining.
- **Calibration is acceptance, not re-tuning**: see [Plug-and-play deployment](#plug-and-play-deployment). Wide plausibility limits cannot detect small sensor offsets.
- **Online thermal window warm-up**: The online predictor buffers the last 5 cycles for the causal thermal effectiveness window; after start-up, a block gap or a server restart the buffer resets, so the next 4 cycles use a shorter window than training.

## Security

- **Read-only nodes**: All OPC-UA sensor nodes enforce `AccessLevel = UserAccessLevel = CurrentRead`. The server installs a custom user manager that never grants `Admin` privileges, rejecting write attempts with `BadNotWritable` or `BadUserAccessDenied`.
- **No write path in the advisor**: The supervisory advisor client is strictly passive, subscribing to telemetry frames without any write, method-call, or node-configuration capabilities.
- **Encrypted transport**: OPC-UA communication defaults to `Basic256Sha256` `SignAndEncrypt` using application certificates generated in `certs/` (or `--cert-dir`). The advisor client pins the server certificate on disk, and credentials are sent inside the encrypted channel.
- **Secrets from environment variables**: All credentials and sensitive endpoints (`HYDRO_OPCUA_USER`, `HYDRO_OPCUA_PASSWORD`, `ERPNEXT_URL`, `ERPNEXT_API_KEY`, `ERPNEXT_API_SECRET`) are loaded from environment variables. No secrets or private keys are stored in code or repository configuration files. The only generated secret is the loopback-replay password in `site/credentials.json` (gitignored).
- **Runtime read-only attestation**: every deployment gate starts a throwaway loopback server and verifies that a client write to a sensor node is rejected; a successful write fails the gate.
- **Integrity manifest**: `python -m hydro manifest build|verify` hashes the drive content; set `HYDRO_MANIFEST_KEY` to add an HMAC signature. This detects corruption and tampering; it is not copy protection.

## License

Proprietary. Copyright (c) 2026 the Owner. All rights reserved. No license is granted except by a separate signed agreement; see [LICENSE](LICENSE). Open-source components keep their own licenses ([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)), and the ZeMA dataset is CC BY 4.0 and must be attributed.

Earlier commits of this repository carried the AGPL-3.0-or-later license. Anyone who received those exact versions under that license keeps the rights it gave for those versions.
