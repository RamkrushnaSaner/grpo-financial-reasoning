"""Shared fixtures. Tests run on CPU with no model download: the tiny LoRA model
and fake tokenizer in ``grpo.testing`` drive the REAL step, training,
evaluation, probe and CLI code paths end to end."""
import pytest
import torch

from grpo.config import from_dict
from grpo.data import FinancialTask
from grpo.testing import tiny_policy

SAMPLE = dict(context="Revenue was $5.0B.", question="What was revenue?",
              answer=5.0, answer_type="dollar", steps=1, qtype="margin")

TINY = dict(model=dict(loader="tiny", lora_dropout=0.0), sampling=dict(max_new_tokens=10),
            training=dict(steps=4, window=2), evaluation=dict(n=3, k=2, show=0, probes=False))


def tiny_config(tmp_path=None, **sections):
    data = {k: dict(v) for k, v in TINY.items()}
    for k, v in sections.items():
        data[k] = {**data.get(k, {}), **v}
    if tmp_path is not None:
        data.setdefault("experiment", {}).update(output_dir=str(tmp_path / "results"), name="t")
    return from_dict(data)


@pytest.fixture
def cfg(tmp_path):
    return tiny_config(tmp_path)


@pytest.fixture
def policy():
    return tiny_policy()


@pytest.fixture(scope="session")
def task():
    return FinancialTask(version=1)


def make_trained(policy, scale: float = 0.5, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        policy.model.lora_A.weight.copy_(torch.randn(policy.model.lora_A.weight.shape, generator=g))
        policy.model.lora_B.weight.copy_(
            torch.randn(policy.model.lora_B.weight.shape, generator=g) * scale)
    return policy
