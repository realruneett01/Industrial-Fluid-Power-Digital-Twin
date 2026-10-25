"""Seeded randomized/property tests over the pre-existing core (codec, assembler, indices)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from hydro.data.schema import CHANNEL_REGISTRY, PHYSICAL_CHANNELS
from hydro.deploy.synthetic import synthetic_cycle
from hydro.eval.metrics import latency_percentiles
from hydro.eval.splits import blocked_time_ordered_kfold
from hydro.online.advisor_client import (
    HEALTHY_CLASSES,
    CycleAssembler,
    health_index,
)
from hydro.online.cmms import ACTIONS, build_work_order
from hydro.online.predictor import CycleHistory, extract_cycle_features
from hydro.online.replay_server import (
    BLOCKS_PER_CYCLE,
    ReadOnlyUserManager,
    decode_block_frame,
    encode_block_frame,
    frame_layout,
)

CLASS_VALUES = {
    "cooler": [3, 20, 100],
    "valve": [73, 80, 90, 100],
    "pump": [0, 1, 2],
    "accumulator": [90, 100, 115, 130],
}


def _probs(rng: np.random.Generator) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for t, classes in CLASS_VALUES.items():
        p = rng.dirichlet(np.ones(len(classes)))
        out[t] = (np.array(classes), p)
    return out


@pytest.mark.parametrize("seed", range(150))
def test_frame_roundtrip(seed):
    rng = np.random.default_rng(seed)
    idx = int(rng.integers(0, 2**31 - 1))
    values = {
        n: rng.normal(size=max(1, CHANNEL_REGISTRY[n].rate_hz // 10)) for n in PHYSICAL_CHANNELS
    }
    frame = encode_block_frame(idx, values)
    got_idx, got = decode_block_frame(frame, PHYSICAL_CHANNELS)
    assert got_idx == idx and set(got) == set(values)
    for n in values:
        np.testing.assert_allclose(got[n], values[n])
    assert len(frame) == 1 + sum(w for _, w in frame_layout(PHYSICAL_CHANNELS))


@pytest.mark.parametrize("extra", [-3, -1, 1, 5])
def test_frame_with_wrong_length_is_rejected(extra):
    values = {n: np.zeros(max(1, CHANNEL_REGISTRY[n].rate_hz // 10)) for n in PHYSICAL_CHANNELS}
    frame = encode_block_frame(7, values)
    frame = frame[:extra] if extra < 0 else frame + [0.0] * extra
    with pytest.raises(ValueError):
        decode_block_frame(frame, PHYSICAL_CHANNELS)


@pytest.mark.parametrize("seed", range(150))
def test_health_index_bounds_and_monotonicity(seed):
    rng = np.random.default_rng(seed)
    probs = _probs(rng)
    hi = health_index(probs)
    assert 0.0 <= hi <= 1.0
    expected = np.mean([p[list(c).index(HEALTHY_CLASSES[t])] for t, (c, p) in probs.items()])
    assert hi == pytest.approx(expected)
    # moving mass from a fault class to the healthy class never lowers HI
    t = list(CLASS_VALUES)[seed % 4]
    classes, p = probs[t]
    h = list(classes).index(HEALTHY_CLASSES[t])
    q = p.copy()
    donor = int(np.argmax(np.where(np.arange(len(q)) == h, -1, q)))
    q[h] += q[donor]
    q[donor] = 0.0
    better = {**probs, t: (classes, q)}
    assert health_index(better) >= hi - 1e-12


@pytest.mark.parametrize(
    "bad",
    [
        np.array([0.5, 0.2, 0.2]),
        np.array([-0.1, 0.6, 0.5]),
        np.array([np.nan, 0.5, 0.5]),
        np.array([]),
    ],
)
def test_health_index_rejects_invalid_distributions(bad):
    with pytest.raises(ValueError):
        health_index({"pump": (np.array([0, 1, 2])[: max(len(bad), 1)], bad)})


def _blocks(cycle_start: int, n: int = BLOCKS_PER_CYCLE):
    """Blocks whose samples all equal their block index, so provenance can be checked."""
    for b in range(cycle_start, cycle_start + n):
        yield (
            b,
            {
                name: np.full(max(1, ch.rate_hz // 10), float(b))
                for name, ch in CHANNEL_REGISTRY.items()
                if not ch.is_virtual
            },
        )


@pytest.mark.parametrize("k", [0, 1, 2, 5, 40])
def test_assembler_complete_cycle(k):
    asm, cycles = CycleAssembler(), []
    for b, v in _blocks(k * BLOCKS_PER_CYCLE):
        c = asm.add_block(b, v)
        if c is not None:
            cycles.append(c)
    assert len(cycles) == 1 and asm.dropped_blocks == 0
    for name, arr in cycles[0].items():
        assert arr.shape == (1, CHANNEL_REGISTRY[name].n_samples)
    ps1 = cycles[0]["PS1"].ravel()
    assert (
        ps1.min() == k * BLOCKS_PER_CYCLE
        and ps1.max() == k * BLOCKS_PER_CYCLE + BLOCKS_PER_CYCLE - 1
    )


@pytest.mark.parametrize("seed", range(150))
def test_assembler_random_stream_only_emits_contiguous_aligned_cycles(seed):
    rng = np.random.default_rng(seed)
    asm = CycleAssembler()
    emitted = []
    idx = int(rng.integers(0, 3)) * BLOCKS_PER_CYCLE + int(rng.integers(0, 900))
    for _ in range(int(rng.integers(3, 2400))):
        r = rng.random()
        if r < 0.004:
            idx += int(rng.integers(2, 50))  # dropped blocks
        elif r < 0.006:
            idx = max(0, idx - int(rng.integers(1, 700)))  # restart / rewind
        elif r < 0.01:
            pass  # duplicate delivery of the same index
        else:
            idx += 1
        vals = {
            n: np.full(max(1, c.rate_hz // 10), float(idx))
            for n, c in CHANNEL_REGISTRY.items()
            if not c.is_virtual
        }
        cyc = asm.add_block(idx, vals)
        if cyc is not None:
            emitted.append(cyc)
    for cyc in emitted:
        ps1 = cyc["PS1"].ravel()
        first = int(ps1[0])
        assert first % BLOCKS_PER_CYCLE == 0
        np.testing.assert_array_equal(
            ps1, np.repeat(np.arange(first, first + BLOCKS_PER_CYCLE), 10).astype(float)
        )
        assert cyc["TS1"].shape == (1, 60)
    assert asm.dropped_blocks >= 0 and asm.discarded_cycles >= 0


@pytest.mark.parametrize("seed", range(60))
def test_features_finite_for_corrupted_inputs(seed):
    rng = np.random.default_rng(seed)
    cyc = synthetic_cycle(seed)
    for name in rng.choice(list(cyc), size=int(rng.integers(0, 5)), replace=False):
        arr = cyc[name].copy()
        pos = rng.integers(0, arr.shape[1], size=int(rng.integers(1, 20)))
        arr[0, pos] = rng.choice([np.nan, np.inf, -np.inf], size=len(pos))
        cyc[name] = arr
    x = extract_cycle_features(cyc, baseline=float(rng.uniform(0.1, 3.0)))
    assert x.dtype == np.float32 and x.shape[0] == 1 and np.isfinite(x).all()


@pytest.mark.parametrize("seed", range(25))
def test_features_are_deterministic(seed):
    a = extract_cycle_features(synthetic_cycle(seed), 1.0)
    b = extract_cycle_features(synthetic_cycle(seed), 1.0)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("missing", PHYSICAL_CHANNELS)
def test_features_reject_missing_channel(missing):
    cyc = synthetic_cycle(0)
    del cyc[missing]
    with pytest.raises(ValueError, match=missing):
        extract_cycle_features(cyc, 1.0)


@pytest.mark.parametrize("maxlen", [1, 2, 3, 5, 8])
def test_history_never_exceeds_maxlen_and_stacks_in_order(maxlen):
    h = CycleHistory(maxlen)
    for i in range(maxlen + 4):
        h.append({"TS1": np.full(60, float(i))})
        assert len(h) == min(i + 1, maxlen)
    stacked = h.stack()["TS1"]
    assert stacked.shape == (maxlen, 60)
    assert list(stacked[:, 0]) == [float(i) for i in range(4, maxlen + 4)]
    h.reset()
    assert len(h) == 0 and h.stack() == {}


@pytest.mark.parametrize("seed", range(100))
def test_user_manager_accepts_only_exact_pairs(seed):
    rng = np.random.default_rng(seed)
    alphabet = list("abcXYZ019-_ ")
    name = "".join(rng.choice(alphabet, size=int(rng.integers(1, 8))))
    pw = "".join(rng.choice(alphabet, size=int(rng.integers(1, 12))))
    mgr = ReadOnlyUserManager({name: pw})
    user = mgr.get_user(None, name, pw)
    assert user is not None and user.role.name != "Admin"
    assert mgr.get_user(None, name, pw + "x") is None
    assert mgr.get_user(None, name + "x", pw) is None
    assert mgr.get_user(None, None, None) is None
    assert mgr.get_user(None, "", "") is None
    assert mgr.get_user(None, name, pw.upper() if pw.upper() != pw else pw + "!") is None


@pytest.mark.parametrize("users", [{"": "x"}, {"u": ""}, {"": ""}])
def test_user_manager_rejects_empty_credentials_config(users):
    with pytest.raises(ValueError):
        ReadOnlyUserManager(users)


@pytest.mark.parametrize("seed", range(60))
def test_work_order_valid_inputs(seed):
    rng = np.random.default_rng(seed)
    fault = list(ACTIONS)[seed % 4]
    sev, metric = int(rng.integers(1, 4)), float(rng.uniform(0, 1))
    wo = build_work_order(f"HYD-{seed}", fault, sev, metric)
    assert wo["hydro_fault_type"] == fault and wo["hydro_severity"] == sev
    assert wo["priority"] in {"Low", "Medium", "High"} and f"HYD-{seed}" in wo["subject"]
    assert math.isclose(wo["hydro_metric"], metric)


@pytest.mark.parametrize("fault", ["", "motor", "PUMP", None, 3])
def test_work_order_rejects_unknown_fault(fault):
    with pytest.raises(ValueError):
        build_work_order("A", fault, 2, 0.5)  # type: ignore[arg-type]


@pytest.mark.parametrize("n", range(2, 102))
@pytest.mark.parametrize("splits", [2, 5])
def test_blocked_kfold_partitions_without_leakage(n, splits):
    if splits > n:
        with pytest.raises(ValueError):
            blocked_time_ordered_kfold(n, splits)
        return
    folds = blocked_time_ordered_kfold(n, splits)
    tests = np.concatenate([t for _, t in folds])
    assert sorted(tests) == list(range(n))  # every cycle tested exactly once
    for train, test in folds:
        assert not set(train) & set(test)
        assert len(train) + len(test) == n
        assert list(test) == list(range(test[0], test[-1] + 1))  # contiguous


@pytest.mark.parametrize("seed", range(40))
def test_latency_percentiles_are_ordered(seed):
    rng = np.random.default_rng(seed)
    samples = rng.integers(1_000_000, 90_000_000, size=int(rng.integers(5, 300)))
    r = latency_percentiles(samples.astype(float))
    vals = list(r.values())
    assert all(v >= 0 and math.isfinite(v) for v in vals)
    p = [r[k] for k in sorted(r) if k.startswith("p")]
    assert p == sorted(p)


def test_synthetic_cycle_is_valid_dataframe_input():
    cyc = synthetic_cycle(3)
    df = pd.DataFrame({k: [float(v.mean())] for k, v in cyc.items()})
    assert df.shape == (1, len(PHYSICAL_CHANNELS)) and np.isfinite(df.to_numpy()).all()
