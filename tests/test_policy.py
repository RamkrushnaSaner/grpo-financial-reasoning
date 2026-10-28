"""Adapter state machine and reference policy: the source of three silent bugs."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from conftest import make_trained

from grpo.algorithm import per_token_logps
from grpo.config import ModelConfig
from grpo.model import MODEL_LOADERS, Policy, load_policy
from grpo.validation import validate_reference_policy

IDS = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])


def test_pristine_policy_is_untrained(policy):
    assert not policy.is_trained() and policy.adapter_magnitude() == 0.0


def test_snapshot_restore_reset(policy):
    make_trained(policy)
    snap = policy.snapshot()
    assert policy.is_trained()
    policy.reset()
    assert not policy.is_trained()
    policy.restore(snap)
    assert policy.is_trained()


def test_use_base_is_reentrant(policy):
    """Regression: peft's disable_adapter() is a single-use generator; entering the
    same object twice raised AttributeError('...has no attribute args')."""
    make_trained(policy)
    cm = policy.use_base()
    with cm:
        assert not policy.is_trained()
    with cm:
        assert not policy.is_trained()
    assert policy.is_trained()


def test_use_base_restores_on_exception(policy):
    make_trained(policy)
    mag = policy.adapter_magnitude()
    with pytest.raises(RuntimeError):
        with policy.use_base():
            raise RuntimeError("boom")
    assert policy.adapter_magnitude() == pytest.approx(mag)


def test_adapter_roundtrip(policy, tmp_path):
    make_trained(policy)
    mag = policy.adapter_magnitude()
    path = str(tmp_path / "a" / "adapter.pt")
    policy.save_adapter(path)
    policy.reset()
    assert policy.load_adapter(path) and policy.adapter_magnitude() == pytest.approx(mag)
    assert not policy.load_adapter(str(tmp_path / "missing.pt"))


def test_reference_matches_policy_when_pristine(policy):
    assert torch.allclose(per_token_logps(policy, IDS, use_adapter=True),
                          per_token_logps(policy, IDS, use_adapter=False))


def test_reference_differs_when_trained_and_adapter_survives(policy):
    """If the reference silently equalled the policy, KL would be identically zero."""
    make_trained(policy)
    mag = policy.adapter_magnitude()
    lp_pol = per_token_logps(policy, IDS, use_adapter=True)
    lp_ref = per_token_logps(policy, IDS, use_adapter=False)
    assert (lp_pol - lp_ref).abs().mean() > 1e-4
    assert policy.adapter_magnitude() == pytest.approx(mag)


def test_validate_reference_policy_both_states(policy):
    assert validate_reference_policy(policy, IDS, verbose=False)["mean_abs_diff"] < 1e-6
    make_trained(policy)
    out = validate_reference_policy(policy, IDS, verbose=False)
    assert out["mean_abs_diff"] > 1e-6 and out["kl"] > 0


def test_model_without_lora_is_rejected():
    with pytest.raises(ValueError, match="LoRA"):
        Policy(nn.Linear(2, 2))


def test_stop_token_ids_merge_tokenizer_and_generation_config(policy):
    assert policy.stop_token_ids() == [1]
    policy.model.generation_config = SimpleNamespace(eos_token_id=[7, 1])
    assert policy.stop_token_ids() == [1, 7]


def test_loaders_are_registered_and_tiny_loader_honours_dropout():
    assert {"hf_lora", "tiny"} <= set(MODEL_LOADERS.names())
    p = load_policy(ModelConfig(loader="tiny", lora_dropout=0.5))
    assert p.model.lora_dropout.p == 0.5
