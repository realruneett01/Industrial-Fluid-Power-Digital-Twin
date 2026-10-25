from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from hydro.deploy import selfcheck as sc
from hydro.deploy.results import Status
from hydro.deploy.synthetic import synthetic_cycle
from hydro.online.predictor import TARGETS


@pytest.fixture(scope="session")
def bundle(tmp_path_factory):
    return sc.toolchain_bundle(tmp_path_factory.mktemp("bundle") / "b")


@pytest.fixture
def bundle_copy(bundle, tmp_path):
    dst = tmp_path / "b"
    shutil.copytree(bundle, dst)
    return dst


def test_valid_bundle_passes_files_and_inference(bundle):
    assert sc.check_bundle_files(bundle).status == Status.PASS
    res = sc.check_inference(bundle, runs=6, sla_ms=1e9)
    assert [r.name for r in res] == ["inference", "latency-sla"]
    assert all(r.status == Status.PASS for r in res)


@pytest.mark.parametrize("victim", list(sc.BUNDLE_FILES))
def test_each_missing_bundle_file_fails(bundle_copy, victim):
    (bundle_copy / victim).unlink()
    res = sc.check_bundle_files(bundle_copy)
    assert res.status == Status.FAIL and victim in res.detail


@pytest.mark.parametrize(
    "content",
    ["", "{", "[]", "{}", '{"valve_baseline_s": 1}', '{"valve_baseline_s": 1, "classes": {}}'],
)
def test_corrupt_bundle_json_fails(bundle_copy, content):
    (bundle_copy / "bundle.json").write_text(content)
    assert sc.check_bundle_files(bundle_copy).status == Status.FAIL


@pytest.mark.parametrize("value", [0, -1.0, "nan", "inf"])
def test_bad_valve_baseline_fails(bundle_copy, value):
    meta = json.loads((bundle_copy / "bundle.json").read_text())
    meta["valve_baseline_s"] = float(value) if isinstance(value, str) else value
    (bundle_copy / "bundle.json").write_text(json.dumps(meta))
    assert sc.check_bundle_files(bundle_copy).status == Status.FAIL


def test_empty_class_list_fails(bundle_copy):
    meta = json.loads((bundle_copy / "bundle.json").read_text())
    meta["classes"]["pump"] = []
    (bundle_copy / "bundle.json").write_text(json.dumps(meta))
    assert sc.check_bundle_files(bundle_copy).status == Status.FAIL


@pytest.mark.parametrize("target", TARGETS)
def test_truncated_onnx_fails_inference(bundle_copy, target):
    f = bundle_copy / f"catboost_{target}.onnx"
    f.write_bytes(f.read_bytes()[:50])
    res = sc.check_inference(bundle_copy, runs=3)
    assert res[0].status == Status.FAIL


def test_class_count_mismatch_fails_inference(bundle_copy):
    meta = json.loads((bundle_copy / "bundle.json").read_text())
    meta["classes"]["cooler"] = meta["classes"]["cooler"][:-1]
    (bundle_copy / "bundle.json").write_text(json.dumps(meta))
    assert sc.check_inference(bundle_copy, runs=3)[0].status == Status.FAIL


def test_latency_sla_failure_is_reported(bundle):
    res = sc.check_inference(bundle, runs=6, sla_ms=1e-6)
    assert res[-1].name == "latency-sla" and res[-1].status == Status.FAIL


@pytest.mark.parametrize("threads", [0, 1, 2])
def test_inference_with_thread_settings(bundle, threads):
    assert sc.check_inference(bundle, runs=4, sla_ms=1e9, threads=threads)[0].status == Status.PASS


@pytest.mark.parametrize("seed", range(10))
def test_predictions_are_deterministic_across_sessions(bundle, seed):
    from hydro.online.predictor import OnnxCyclePredictor

    a = OnnxCyclePredictor.from_dir(bundle)(synthetic_cycle(seed))
    b = OnnxCyclePredictor.from_dir(bundle, 1)(synthetic_cycle(seed))
    for t in TARGETS:
        np.testing.assert_allclose(a[t][1], b[t][1], rtol=1e-5, atol=1e-6)


def test_read_only_boundary_holds():
    res = sc.check_read_only_boundary()
    assert res.status == Status.PASS, res.detail


def test_read_only_attestation_detects_a_broken_boundary(monkeypatch):
    """Mutation test: if the server granted Admin, asyncua would let writes through."""
    from asyncua.crypto.permission_rules import User, UserRole

    from hydro.online import replay_server

    def admin(self, iserver, username=None, password=None, certificate=None):
        return User(role=UserRole.Admin, name=username)

    monkeypatch.setattr(replay_server.ReadOnlyUserManager, "get_user", admin)
    res = sc.check_read_only_boundary()
    assert res.status == Status.FAIL and "SUCCEEDED" in res.detail


def test_toolchain_smoke():
    assert sc.check_toolchain().status == Status.PASS
