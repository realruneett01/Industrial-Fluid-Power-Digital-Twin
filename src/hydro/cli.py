"""Command line entry point: python -m hydro <ingest|eval|replay>."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import yaml

from hydro.paths import project_root

REPO_ROOT = project_root()


def _load_config(path: str) -> dict:
    with open(REPO_ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_ingest(args: argparse.Namespace) -> int:
    from hydro.data.download import fetch_zema_dataset
    from hydro.data.loader import load_cycle_dataset, save_interim

    raw = fetch_zema_dataset(Path(args.data_dir) / "raw")
    data, targets = load_cycle_dataset(raw)
    for path in save_interim(data, targets, Path(args.data_dir) / "interim"):
        print(f"wrote {path}")
    return 0


def _load_eval_inputs(cfg: dict):
    from hydro.data.loader import load_cycle_dataset
    from hydro.eval.splits import audit_class_coverage, blocked_time_ordered_kfold

    data, targets = load_cycle_dataset(REPO_ROOT / cfg["paths"]["raw"], cfg["exclude_virtual"])
    folds = blocked_time_ordered_kfold(len(targets), cfg["split"]["n_splits"])
    return data, targets, folds, audit_class_coverage(targets, folds)


def _check_tables(tables: dict, report: Path) -> None:
    """--heavy-check: report exists and every metric is finite."""
    if not report.is_file() or report.stat().st_size == 0:
        raise SystemExit(f"heavy-check: report {report} missing or empty")
    for name, table in tables.items():
        values = table.select_dtypes("number").to_numpy().ravel()
        if not all(math.isfinite(v) for v in values):
            raise SystemExit(f"heavy-check: non-finite metric in table {name}")


def cmd_eval(args: argparse.Namespace) -> int:
    from hydro.eval import benchmark as bm

    cfg = _load_config(args.config)
    data, targets, folds, coverage = _load_eval_inputs(cfg)
    notes = {"Class coverage audit (blocked folds)": str(coverage)}
    if args.split == "blocked":
        tables = {"Baseline (LDA, RF) - blocked 5-fold": bm.run_baseline(data, targets, folds)}
        if args.mode == "full-comparison":
            tables["Four-way comparison - blocked 5-fold"] = bm.run_four_way_benchmark(
                data, targets, folds
            )
        report = REPO_ROOT / cfg["paths"]["reports"] / "baseline_results.md"
    else:
        table, threshold = bm.run_temperature_shift_experiment(
            data, targets, cfg["split"]["temp_shift_threshold_c"]
        )
        tables = {"Temperature-shift generalization": table}
        notes["Temperature threshold (TS1 cycle mean)"] = f"{threshold:.2f} degC"
        if args.mode == "full-comparison":
            tables["Four-way comparison - blocked 5-fold"] = bm.run_four_way_benchmark(
                data, targets, folds
            )
        report = REPO_ROOT / cfg["paths"]["reports"] / "benchmark_results.md"
    bm.write_benchmark_report(tables, report, notes)
    print(f"wrote {report}")
    if args.heavy_check:
        _check_tables(tables, report)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from hydro.online.replay_server import main as replay_main

    replay_main(["--data-dir", args.data_dir, "--cycles", str(args.cycles)])
    return 0


def cmd_bundle(args: argparse.Namespace) -> int:
    if args.holdout < 0:
        raise ValueError(f"Holdout ({args.holdout}) cannot be negative")

    import numpy as np

    from hydro.data.loader import load_cycle_dataset
    from hydro.online.predictor import train_onnx_bundle

    data, targets = load_cycle_dataset(Path(args.data_dir))
    n = len(targets)
    if n <= args.holdout:
        raise ValueError(f"Holdout ({args.holdout}) must be less than cycle count ({n})")
    train_idx = np.arange(n - args.holdout)
    out_dir = train_onnx_bundle(
        data, targets, train_idx, Path(args.out), iterations=args.iterations
    )
    print(out_dir)
    return 0


def cmd_cmms_setup(args: argparse.Namespace) -> int:
    from hydro.online.cmms import ensure_erpnext_schema

    try:
        url = args.url or os.environ.get("ERPNEXT_URL")
        counts = ensure_erpnext_schema(base_url=url)
        print(counts)
        return 0
    except Exception as err:  # comment: catch setup errors and return non-zero exit code
        print(f"CMMS setup failed: {err}")
        return 1


def _site_dir(args: argparse.Namespace) -> Path:
    from hydro.deploy.site import default_site_dir

    return Path(args.site_dir) if args.site_dir else default_site_dir(REPO_ROOT)


def cmd_doctor(args: argparse.Namespace) -> int:
    from hydro.deploy.doctor import run_doctor
    from hydro.deploy.results import Status, render, worst

    results = run_doctor(_site_dir(args))
    print(render(results))
    return 2 if worst(results) == Status.FAIL else 0


def cmd_discover(args: argparse.Namespace) -> int:
    import asyncio
    import json

    from hydro.deploy.discover import discover

    found = asyncio.run(discover(REPO_ROOT, [args.endpoint] if args.endpoint else None))
    print(json.dumps(found.to_dict(), indent=2))
    return 0


def cmd_deploy(args: argparse.Namespace) -> int:
    import asyncio

    from hydro.deploy.gate import GateOptions, run_gate

    opts = GateOptions(
        root=REPO_ROOT,
        models_dir=Path(args.models_dir) if args.models_dir else None,
        site_dir=Path(args.site_dir) if args.site_dir else None,
        mode=args.mode,
        endpoint=args.endpoint,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        insecure=args.insecure,
        strict=args.strict,
        calibration_cycles=args.calibration_cycles,
        toolchain=args.toolchain,
        replay_speed=args.replay_speed,
    )
    try:
        code, report = run_gate(opts)
    except ValueError as err:
        print(f"hydro deploy: {err}")
        return 1
    print(report["text"])
    print(f"\nVERDICT: {report['verdict']} (gate {'passed' if code == 0 else 'FAILED'})")
    if code != 0 or not args.start:
        return code
    from hydro.deploy.service import serve

    return asyncio.run(serve(_site_dir(args)))


def cmd_run(args: argparse.Namespace) -> int:
    import asyncio

    from hydro.deploy.service import serve

    try:
        return asyncio.run(serve(_site_dir(args), args.duration_sec))
    except ValueError as err:
        print(f"hydro run: {err}")
        return 1


def cmd_health(args: argparse.Namespace) -> int:
    from hydro.deploy.service import health

    ok, msg = health(_site_dir(args), args.max_age)
    print(msg)
    return 0 if ok else 1


def cmd_manifest(args: argparse.Namespace) -> int:
    from hydro.deploy import manifest

    key = os.environ.get(manifest.KEY_ENV)
    if args.action == "build":
        print(manifest.write_manifest(REPO_ROOT, key))
        return 0
    result = manifest.verify_manifest(REPO_ROOT, key)
    print({k: v for k, v in result.items() if v or k == "ok"})
    return 0 if result["ok"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hydro")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="download, verify and partition the dataset")
    ingest.add_argument("--data-dir", default="data")
    ingest.set_defaults(func=cmd_ingest)
    ev = sub.add_parser("eval", help="run benchmarks and write a markdown report")
    ev.add_argument("--config", default="configs/data.yaml")
    ev.add_argument("--split", choices=["blocked", "temp-shift"], default="blocked")
    ev.add_argument("--mode", choices=["baseline-only", "full-comparison"], default="baseline-only")
    ev.add_argument("--heavy-check", action="store_true")
    ev.set_defaults(func=cmd_eval)
    rp = sub.add_parser("replay", help="serve cycles over read-only OPC-UA")
    rp.add_argument("--data-dir", default="data/raw")
    rp.add_argument("--cycles", type=int, default=1)
    rp.set_defaults(func=cmd_replay)
    bd = sub.add_parser("bundle", help="train and export online model bundle")
    bd.add_argument("--data-dir", default="data/raw")
    bd.add_argument("--out", default="models/online")
    bd.add_argument("--holdout", type=int, default=205)
    bd.add_argument("--iterations", type=int, default=None)
    bd.set_defaults(func=cmd_bundle)
    cs = sub.add_parser("cmms-setup", help="ensure ERPNext priorities and custom fields exist")
    cs.add_argument(
        "--url", default=os.environ.get("ERPNEXT_URL"), help="ERPNext URL (default: ERPNEXT_URL)"
    )
    cs.set_defaults(func=cmd_cmms_setup)
    site_kw = {"default": None, "help": "site folder (default: <root>/site or $HYDRO_SITE_DIR)"}
    dr = sub.add_parser("doctor", help="check this machine can run hydro")
    dr.add_argument("--site-dir", **site_kw)
    dr.set_defaults(func=cmd_doctor)
    dc = sub.add_parser("discover", help="list reachable sensor sources")
    dc.add_argument("--endpoint", default=None)
    dc.set_defaults(func=cmd_discover)
    dp = sub.add_parser("deploy", help="preflight, calibrate, self-check; optionally start")
    dp.add_argument("--mode", choices=["auto", "live", "replay"], default="auto")
    dp.add_argument("--endpoint", default=None)
    dp.add_argument("--data-dir", default=None)
    dp.add_argument("--models-dir", default=None)
    dp.add_argument("--site-dir", **site_kw)
    dp.add_argument("--insecure", action="store_true", help="unencrypted OPC-UA (lab only)")
    dp.add_argument("--strict", action="store_true", help="treat sensor warnings as failures")
    dp.add_argument("--calibration-cycles", type=int, default=None)
    dp.add_argument("--replay-speed", type=float, default=1.0, help="replay mode only")
    dp.add_argument("--toolchain", action="store_true", help="also train/export a smoke bundle")
    dp.add_argument("--start", action="store_true", help="start the service if the gate passes")
    dp.set_defaults(func=cmd_deploy)
    rn = sub.add_parser("run", help="start the service from a passed deployment")
    rn.add_argument("--site-dir", **site_kw)
    rn.add_argument("--duration-sec", type=float, default=0.0, help="0 = run until stopped")
    rn.set_defaults(func=cmd_run)
    hl = sub.add_parser("health", help="exit 0 when the running service heartbeat is fresh")
    hl.add_argument("--site-dir", **site_kw)
    hl.add_argument("--max-age", type=float, default=120.0)
    hl.set_defaults(func=cmd_health)
    mf = sub.add_parser("manifest", help="build or verify the drive integrity manifest")
    mf.add_argument("action", choices=["build", "verify"])
    mf.set_defaults(func=cmd_manifest)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
