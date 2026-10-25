"""The deployment gate: doctor -> manifest -> discover -> calibrate -> self-check -> verdict.

Nothing is started unless every step is PASS or WARN. The full evidence is written to
``site/deploy_report.json`` and the machine-specific configuration to ``site/site.json``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hydro.deploy import calibrate, discover, doctor, manifest, selfcheck, sources
from hydro.deploy.machine import auto_profile, detect_machine
from hydro.deploy.results import CheckResult, Status, render, worst
from hydro.deploy.site import (
    CALIBRATION_FILE,
    REPORT_FILE,
    SITE_FILE,
    SITE_VERSION,
    default_site_dir,
    write_json_atomic,
)
from hydro.online import security
from hydro.online.replay_server import DEFAULT_ENDPOINT

EXIT_OK, EXIT_GATE_FAILED = 0, 2


@dataclass
class GateOptions:
    root: Path
    models_dir: Path | None = None
    site_dir: Path | None = None
    mode: str = "auto"  # auto | live | replay
    endpoint: str | None = None
    data_dir: Path | None = None
    cert_dir: Path | None = None
    insecure: bool = False
    strict: bool = False
    calibration_cycles: int | None = None
    toolchain: bool = False
    capture_timeout_s: float | None = None
    replay_speed: float = 1.0


def _pick_source(opts: GateOptions, found: discover.Discovery, steps: list[CheckResult]):
    """Choose live OPC-UA or dataset replay. Returns (kind, endpoint, data_dir) or None."""
    reachable = found.reachable_endpoints()
    if opts.mode in ("auto", "live") and (opts.endpoint or reachable):
        endpoint = opts.endpoint or reachable[0]["url"]
        steps.append(CheckResult("source", Status.PASS, f"live OPC-UA at {endpoint}"))
        return "live", endpoint, None
    if opts.mode == "live":
        steps.append(
            CheckResult(
                "source",
                Status.FAIL,
                "no reachable OPC-UA endpoint",
                "pass --endpoint opc.tcp://host:port/path or set HYDRO_OPCUA_ENDPOINTS",
            )
        )
        return None
    data_dir = opts.data_dir or (Path(found.datasets[0]) if found.datasets else None)
    if data_dir is not None:
        steps.append(
            CheckResult(
                "source",
                Status.WARN,
                f"no live sensors found; using recorded dataset {data_dir} (REPLAY, not live)",
                "connect the OPC-UA gateway to run on real sensors",
            )
        )
        return "replay", DEFAULT_ENDPOINT, data_dir
    steps.append(
        CheckResult(
            "source",
            Status.FAIL,
            "no sensor source: no OPC-UA endpoint reachable and no dataset on the drive",
            "connect the sensor gateway, or copy a ZeMA dataset folder onto the drive",
        )
    )
    return None


def _calibrate(opts, kind, endpoint, data_dir, n, steps, previous):
    """Capture cycles from the chosen source and judge them. Returns (baseline, source_label)."""
    try:
        if kind == "live":
            user = os.environ.get("HYDRO_OPCUA_USER", "")
            password = os.environ.get("HYDRO_OPCUA_PASSWORD", "")
            if not user or not password:
                steps.append(
                    CheckResult(
                        "credentials",
                        Status.FAIL,
                        "HYDRO_OPCUA_USER / HYDRO_OPCUA_PASSWORD not set",
                        "export both variables (never store them on the drive)",
                    )
                )
                return {}, "live"
            if not opts.insecure:
                server_cert = security.cert_pair(opts.cert_dir, "server").cert
                if not server_cert.is_file():
                    steps.append(
                        CheckResult(
                            "server-certificate",
                            Status.FAIL,
                            f"server certificate not found at {server_cert}",
                            "copy the OPC-UA server's certificate there as server_cert.der "
                            "(or use --insecure on an isolated lab network)",
                        )
                    )
                    return {}, "live"
            timeout = opts.capture_timeout_s or sources.expected_capture_seconds(n) + 30
            cycles = asyncio.run(
                sources.grab_from_opcua(
                    endpoint, n, timeout, user, password, None if opts.insecure else opts.cert_dir
                )
            )
            label = f"live OPC-UA {endpoint}"
        else:
            cycles = sources.grab_from_dataset(data_dir, n)
            label = f"dataset {data_dir}"
    except Exception as err:  # noqa: BLE001  # connection/IO problems are gate failures
        steps.append(CheckResult("capture", Status.FAIL, f"{type(err).__name__}: {err}"))
        return {}, ""
    if len(cycles) < n:
        steps.append(
            CheckResult(
                "capture",
                Status.FAIL,
                f"captured {len(cycles)} of {n} complete cycles",
                "check the stream is running and not dropping blocks",
            )
        )
        return {}, label
    steps.append(CheckResult("capture", Status.PASS, f"{len(cycles)} complete cycle(s)"))
    results, baseline = calibrate.calibrate_cycles(cycles, opts.strict)
    steps.extend(results)
    if previous:
        steps.extend(calibrate.drift_report(previous, cycles[0]))
    return baseline, label


def run_gate(opts: GateOptions) -> tuple[int, dict[str, Any]]:
    """Run every step; returns (exit_code, report). Always writes report and site files."""
    if opts.replay_speed <= 0:
        raise ValueError(f"replay_speed must be positive, got {opts.replay_speed}")
    root = opts.root.resolve()
    site_dir = (opts.site_dir or default_site_dir(root)).resolve()
    models_dir = (opts.models_dir or root / "models" / "online").resolve()
    cert_dir = opts.cert_dir or site_dir / "certs"
    opts.cert_dir = cert_dir
    machine = detect_machine()
    profile = auto_profile(machine)
    steps: list[CheckResult] = list(doctor.run_doctor(site_dir, machine))

    check = manifest.verify_manifest(root, os.environ.get(manifest.KEY_ENV))
    if "error" in check:
        steps.append(
            CheckResult(
                "manifest", Status.WARN, check["error"], "run `hydro manifest build` when packing"
            )
        )
    elif not check["ok"]:
        bad = check["missing"] + check["changed"] or ["signature"]
        steps.append(
            CheckResult(
                "manifest",
                Status.FAIL,
                f"drive content differs from manifest: {', '.join(bad[:5])}",
                "re-copy the drive from the original; do not deploy tampered files",
            )
        )
    else:
        steps.append(CheckResult("manifest", Status.PASS, f"signature {check['signature']}"))

    found = asyncio.run(discover.discover(root, [opts.endpoint] if opts.endpoint else None))
    chosen = _pick_source(opts, found, steps)
    previous: dict[str, Any] = {}
    with contextlib.suppress(ValueError):  # first run or unreadable old file: nothing to compare
        previous = calibrate.load_calibration(site_dir / CALIBRATION_FILE)["channels"]
    baseline: dict[str, Any] = {}
    kind, endpoint, data_dir, label = "none", "", None, ""
    if chosen is not None:
        kind, endpoint, data_dir = chosen
        default_n = profile["calibration_cycles"]
        n = opts.calibration_cycles or (min(default_n, 2) if kind == "live" else default_n)
        baseline, label = _calibrate(opts, kind, endpoint, data_dir, n, steps, previous)

    bundle = selfcheck.check_bundle_files(models_dir)
    steps.append(bundle)
    if bundle.status != Status.FAIL:
        steps.extend(
            selfcheck.check_inference(
                models_dir,
                sla_ms=profile["latency_sla_ms"],
                threads=profile["onnx_intra_op_threads"],
            )
        )
    steps.append(selfcheck.check_read_only_boundary())
    if opts.toolchain:
        steps.append(selfcheck.check_toolchain())

    verdict = worst(steps)
    passed = verdict != Status.FAIL
    created = datetime.now(UTC).isoformat(timespec="seconds")
    record = calibrate.build_calibration_record(baseline, label, steps) if baseline else None
    if record is not None:
        calibrate.save_calibration(record, site_dir / CALIBRATION_FILE)
    report = {
        "created": created,
        "verdict": verdict.name,
        "gate_passed": passed,
        "machine": machine.to_dict(),
        "profile": profile,
        "discovery": found.to_dict(),
        "steps": [s.to_dict() for s in steps],
    }
    write_json_atomic(site_dir / REPORT_FILE, report)
    site = {
        "version": SITE_VERSION,
        "created": created,
        "gate_passed": passed,
        "source": kind,
        "endpoint": endpoint,
        "data_dir": str(data_dir) if data_dir else "",
        "models_dir": str(models_dir),
        "cert_dir": str(cert_dir),
        "insecure": opts.insecure,
        "replay_speed": opts.replay_speed,
        "profile": profile,
        "calibration": str(site_dir / CALIBRATION_FILE) if record else "",
    }
    write_json_atomic(site_dir / SITE_FILE, site)
    report["text"] = render(steps)
    return (EXIT_OK if passed else EXIT_GATE_FAILED), report
