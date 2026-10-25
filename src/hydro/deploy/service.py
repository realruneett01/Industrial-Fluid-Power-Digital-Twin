"""Long-running supervised service started after the gate passes."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from hydro.deploy.site import (
    ALERTS_FILE,
    heartbeat_age,
    load_site,
    loopback_credentials,
    write_heartbeat,
)

log = logging.getLogger("hydro.service")
BACKOFF_START_S, BACKOFF_MAX_S = 2.0, 60.0
HEARTBEAT_EVERY_S = 10.0


def make_dispatch(site_dir: Path):
    """CMMS dispatch when ERPNext is configured, otherwise a local JSONL alert log."""
    from hydro.online.cmms import dispatch_cmms_work_order

    if os.environ.get("ERPNEXT_URL"):
        return dispatch_cmms_work_order
    path = site_dir / ALERTS_FILE

    def to_file(payload: dict[str, Any]) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.time(), **payload}) + "\n")
        return 0

    return to_file


def _beat(site_dir: Path, client) -> None:
    hist = client.hi_history
    write_heartbeat(
        site_dir,
        cycles=len(hist),
        last_hi=hist[-1] if hist else None,
        cycle_errors=client.cycle_errors,
        bad_frames=client.bad_frames,
        dropped_blocks=client.assembler.dropped_blocks,
    )


async def _heartbeat(site_dir: Path, client) -> None:
    while True:
        _beat(site_dir, client)
        await asyncio.sleep(HEARTBEAT_EVERY_S)


async def _run_once(site: dict[str, Any], site_dir: Path, duration_s: float) -> None:
    from hydro.online.advisor_client import AdvisorConfig, SupervisoryAdvisorClient
    from hydro.online.predictor import OnnxCyclePredictor
    from hydro.online.replay_server import HydroReplayServer

    profile = site["profile"]
    predictor = OnnxCyclePredictor.from_dir(site["models_dir"], profile["onnx_intra_op_threads"])
    insecure = bool(site.get("insecure")) or site["source"] == "replay"
    cert_dir = None if insecure else Path(site["cert_dir"])
    server = None
    replay_task = None
    if site["source"] == "replay":
        from hydro.data.loader import load_cycle_dataset

        creds = loopback_credentials(site_dir)
        user, password = creds["user"], creds["password"]
        data, _ = load_cycle_dataset(Path(site["data_dir"]))
        server = HydroReplayServer(data, site["endpoint"], {user: password}, cert_dir=None)
        await server.start()
        total = len(next(iter(data.values())))

        async def loop_cycles() -> None:
            while True:
                await server.run(range(total), float(site.get("replay_speed", 1.0)))

        replay_task = asyncio.create_task(loop_cycles())
    else:
        user, password = (
            os.environ.get("HYDRO_OPCUA_USER", ""),
            os.environ.get("HYDRO_OPCUA_PASSWORD", ""),
        )
    client = SupervisoryAdvisorClient(
        site["endpoint"],
        user,
        password,
        predictor,
        AdvisorConfig(
            publish_ms=profile["opcua_publish_ms"],
            dispatch=make_dispatch(site_dir),
            cert_dir=cert_dir,
        ),
    )
    beat = asyncio.create_task(_heartbeat(site_dir, client))
    try:
        await client.run(duration_s)
    finally:
        beat.cancel()
        _beat(site_dir, client)
        if replay_task is not None:
            replay_task.cancel()
        if server is not None:
            await server.stop()


async def serve(site_dir: Path, duration_s: float = 0.0, max_restarts: int = 0) -> int:
    """Run until cancelled (duration 0) or for duration_s; reconnect with capped backoff.

    ``max_restarts`` > 0 stops after that many failed attempts; 0 retries forever, except that
    a bounded run (duration_s > 0) makes a single attempt, so scripts and tests cannot hang.
    """
    site = load_site(site_dir)
    if duration_s > 0 and max_restarts == 0:
        max_restarts = 1
    delay, failures = BACKOFF_START_S, 0
    while True:
        started = time.monotonic()
        try:
            await _run_once(site, site_dir, duration_s or 10**9)
            return 0
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("service attempt failed")
            failures += 1
            if max_restarts and failures >= max_restarts:
                return 1
            if time.monotonic() - started > 5 * BACKOFF_MAX_S:
                delay = BACKOFF_START_S  # ran for a while: treat as a fresh failure series
            await asyncio.sleep(delay)
            delay = min(delay * 2, BACKOFF_MAX_S)


def health(site_dir: Path, max_age_s: float) -> tuple[bool, str]:
    """Container/systemd health: is the heartbeat fresh?"""
    age = heartbeat_age(site_dir)
    if age is None:
        return False, "no heartbeat yet"
    return age <= max_age_s, f"heartbeat {age:.0f}s old"
