"""Command line entry point: python -m hydro <ingest|eval|replay>."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
