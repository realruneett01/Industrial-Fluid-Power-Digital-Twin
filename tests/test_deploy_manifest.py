from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from hydro.deploy import manifest


def _make_tree(root: Path, n: int = 6, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    names = []
    for i in range(n):
        sub = root / f"d{i % 3}"
        sub.mkdir(exist_ok=True)
        f = sub / f"f{i}.bin"
        f.write_bytes(rng.randbytes(rng.randint(1, 4000)))
        names.append(f.relative_to(root).as_posix())
    return names


def test_build_verify_roundtrip(tmp_path):
    names = _make_tree(tmp_path)
    manifest.write_manifest(tmp_path)
    res = manifest.verify_manifest(tmp_path)
    assert res["ok"] and not res["missing"] and not res["changed"] and not res["extra"]
    assert res["signature"] == "unsigned"
    assert set(json.loads((tmp_path / manifest.MANIFEST_NAME).read_text())["files"]) == set(names)


def test_missing_manifest_reports_error(tmp_path):
    res = manifest.verify_manifest(tmp_path)
    assert not res["ok"] and "not found" in res["error"]


@pytest.mark.parametrize("content", ["", "{", "[]", '{"files": []}', '{"files": 3}', "null"])
def test_corrupt_manifest_is_not_ok_and_does_not_raise(tmp_path, content):
    (tmp_path / manifest.MANIFEST_NAME).write_text(content)
    res = manifest.verify_manifest(tmp_path)
    assert not res["ok"] and "error" in res


@pytest.mark.parametrize("seed", range(120))
def test_single_byte_flip_is_detected(tmp_path, seed):
    names = _make_tree(tmp_path, seed=seed)
    manifest.write_manifest(tmp_path)
    rng = random.Random(seed)
    target = tmp_path / rng.choice(names)
    data = bytearray(target.read_bytes())
    data[rng.randrange(len(data))] ^= 1 << rng.randrange(8)
    target.write_bytes(bytes(data))
    res = manifest.verify_manifest(tmp_path)
    assert not res["ok"] and res["changed"] == [target.relative_to(tmp_path).as_posix()]


@pytest.mark.parametrize("seed", range(40))
def test_deleted_file_is_detected(tmp_path, seed):
    names = _make_tree(tmp_path, seed=seed)
    manifest.write_manifest(tmp_path)
    victim = random.Random(seed).choice(names)
    (tmp_path / victim).unlink()
    res = manifest.verify_manifest(tmp_path)
    assert not res["ok"] and res["missing"] == [victim]


def test_extra_file_is_reported_but_not_fatal(tmp_path):
    _make_tree(tmp_path)
    manifest.write_manifest(tmp_path)
    (tmp_path / "new.txt").write_text("x")
    res = manifest.verify_manifest(tmp_path)
    assert res["ok"] and res["extra"] == ["new.txt"]


@pytest.mark.parametrize(
    "excluded",
    ["site/x.json", "certs/k.pem", "__pycache__/a.pyc", "a.log", ".git/HEAD", "venv/bin/p"],
)
def test_generated_files_are_excluded(tmp_path, excluded):
    _make_tree(tmp_path)
    manifest.write_manifest(tmp_path)
    p = tmp_path / excluded
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("generated")
    res = manifest.verify_manifest(tmp_path)
    assert res["ok"] and not res["extra"]


def test_signature_valid_wrong_key_and_missing_key(tmp_path):
    _make_tree(tmp_path)
    manifest.write_manifest(tmp_path, key="secret")
    assert manifest.verify_manifest(tmp_path, "secret")["signature"] == "valid"
    assert manifest.verify_manifest(tmp_path, "secret")["ok"]
    bad = manifest.verify_manifest(tmp_path, "other")
    assert bad["signature"] == "invalid" and not bad["ok"]
    assert manifest.verify_manifest(tmp_path)["signature"] == "unchecked"


def test_regenerated_manifest_without_key_fails_signed_check(tmp_path):
    names = _make_tree(tmp_path)
    manifest.write_manifest(tmp_path, key="secret")
    (tmp_path / names[0]).write_bytes(b"evil")
    manifest.write_manifest(tmp_path)  # attacker rebuilds an unsigned manifest
    res = manifest.verify_manifest(tmp_path, "secret")
    assert not res["ok"] and res["signature"] == "invalid"


def test_path_traversal_entry_is_treated_as_missing(tmp_path):
    root = tmp_path / "drive"
    root.mkdir()
    (tmp_path / "outside.txt").write_text("x")
    (root / manifest.MANIFEST_NAME).write_text(
        json.dumps({"files": {"../outside.txt": {"sha256": "0" * 64, "size": 1}}})
    )
    res = manifest.verify_manifest(root)
    assert not res["ok"] and res["missing"] == ["../outside.txt"]


def test_symlinks_are_skipped(tmp_path):
    _make_tree(tmp_path)
    (tmp_path / "link").symlink_to(tmp_path / "d0" / "f0.bin")
    assert "link" not in manifest.build_manifest(tmp_path)["files"]


def test_manifest_is_deterministic(tmp_path):
    _make_tree(tmp_path)
    a, b = manifest.build_manifest(tmp_path), manifest.build_manifest(tmp_path)
    assert a["files"] == b["files"]
