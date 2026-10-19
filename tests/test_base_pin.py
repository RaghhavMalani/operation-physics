"""D78: the pinned 4-bit base for Phase 12. Loaders get the snapshot dir of the pinned commit, its weights are
hash-checked before loading, an adapter must name that dir, and the loaded model must come from it."""

import json
from types import SimpleNamespace

import pytest

from physcal import base as pb
from physcal.config import REPO_ROOT

COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """A repo-local-style HF cache holding one snapshot of a fake repo; base_config() returns its pin."""
    from physcal.eval.fingerprint import _dir_content_hash

    hub = tmp_path / "hub"
    snap = hub / "models--org--tiny-bnb-4bit" / "snapshots" / COMMIT
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"weights")
    (hub / "models--org--tiny-bnb-4bit" / "refs").mkdir()
    (hub / "models--org--tiny-bnb-4bit" / "refs" / "main").write_text(COMMIT)
    cfg = {"repo_id": "org/tiny-bnb-4bit", "revision": COMMIT, "content_sha256": _dir_content_hash(snap),
           "model": {"max_seq_length": 64, "text_only": False, "chat_template_kwargs": {}}}
    monkeypatch.setattr(pb, "HF_HUB", hub)
    monkeypatch.setattr(pb, "base_config", lambda name="base_4bit": cfg)
    return snap, cfg


def _adapter(d, base):
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(base)}))
    return d


def test_the_pinned_dir_is_the_commit_snapshot_and_its_weights_are_checked(fake_hub):
    snap, cfg = fake_hub
    d = pb.pinned_base_dir(cfg)
    assert pb._same_dir(d, snap)
    assert pb.verify_base_dir(d, cfg)["content_sha256"] == cfg["content_sha256"]
    (snap / "model.safetensors").write_bytes(b"other weights")  # same commit dir, different weights
    with pytest.raises(pb.BaseNotPinned, match="weights hash"):
        pb.verify_base_dir(snap, cfg)
    with pytest.raises(pb.BaseNotPinned, match="not the snapshot"):
        pb.verify_base_dir(snap.parent, cfg)


def test_a_missing_snapshot_is_never_resolved_to_main(fake_hub):
    _, cfg = fake_hub
    other = dict(cfg, revision="f" * 40)  # not in the cache; offline: raise, do not fall back to refs/main
    with pytest.raises(Exception):
        pb.pinned_base_dir(other)


def test_model_sources(fake_hub, tmp_path):
    snap, cfg = fake_hub
    assert pb.check_model_source(str(snap), cfg)["adapter"] is None
    good = _adapter(tmp_path / "sft_adapter", snap)
    assert pb.check_model_source(str(good), cfg)["adapter"] == str(good)
    for name, base in (("hub_name", "org/tiny-bnb-4bit"), ("other_dir", tmp_path)):
        with pytest.raises(pb.BaseNotPinned):
            pb.check_model_source(str(_adapter(tmp_path / name, base)), cfg)
    with pytest.raises(pb.BaseNotPinned):
        pb.check_model_source(str(tmp_path), cfg)


def test_resolve_model(fake_hub, tmp_path):
    snap, cfg = fake_hub
    assert pb.resolve_model({"id": "Qwen/x", "revision": "abc"}) == ("Qwen/x", "abc", None)  # legacy, unchanged
    name, rev, pin = pb.resolve_model({"pinned_base": "base_4bit"})
    assert pb._same_dir(name, snap) and rev is None and pin["revision"] == COMMIT and pin["config"] == "base_4bit"
    ad = _adapter(tmp_path / "sft_adapter", snap)
    assert pb.resolve_model({"pinned_base": "base_4bit", "id": str(ad)})[0] == str(ad)
    with pytest.raises(ValueError, match="no revision"):
        pb.resolve_model({"pinned_base": "base_4bit", "revision": COMMIT})


def test_assert_loaded_base(fake_hub):
    snap, cfg = fake_hub
    plain = SimpleNamespace(config=SimpleNamespace(_name_or_path=str(snap)))
    assert pb.assert_loaded_base(plain, cfg)
    peft_like = SimpleNamespace(get_base_model=lambda: plain)
    assert pb.assert_loaded_base(peft_like, cfg)
    hub = SimpleNamespace(config=SimpleNamespace(_name_or_path="org/tiny-bnb-4bit"))
    with pytest.raises(pb.BaseNotPinned):
        pb.assert_loaded_base(hub, cfg)


def test_eval_backend_refuses_an_adapter_on_another_base_before_loading(fake_hub, tmp_path):
    from physcal.eval.run_eval import HFBackend

    snap, _ = fake_hub
    ad = _adapter(tmp_path / "rl_adapter", "unsloth/some-other-base")
    with pytest.raises(pb.BaseNotPinned):
        HFBackend(str(snap), {"pinned_base": "base_4bit", "lora": str(ad), "loader": "transformers",
                              "merge_adapter": True, "device": "cpu", "dtype": "float32"}, {})


def test_the_real_pin_matches_the_cached_snapshot():
    """The committed config pins the snapshot the existing SFT adapter was trained on (D70), if it is cached here."""
    cfg = pb.base_config()
    snap = pb.HF_HUB / "models--unsloth--Qwen2.5-Math-1.5B-Instruct-bnb-4bit" / "snapshots" / cfg["revision"]
    if not snap.exists():
        pytest.skip("the pinned snapshot is not in the repo-local HF cache")
    d = pb.pinned_base_dir(cfg)
    assert (REPO_ROOT / d).resolve() == snap.resolve()
    sft = json.loads((REPO_ROOT / "results/sft/qwen2.5-math-1.5b-instruct/adapter/adapter_config.json").read_text())
    assert sft["base_model_name_or_path"] == cfg["repo_id"]  # trained on this repo (at its only cached commit)
