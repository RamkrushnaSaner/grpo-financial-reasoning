"""The policy: a base model with LoRA adapters, plus an explicit adapter-state
machine, and the registered model loaders.

LoRA initialises B = 0, so at step 0 the adapted model is identical to the
base model. Copying that pristine snapshot back in therefore IS the base model,
which gives a free reference policy and a free reset.

Why explicit tensor swapping instead of peft's ``disable_adapter()``: that
context manager is a single-use generator; entering the same object twice
raises ``AttributeError: '_GeneratorContextManager' object has no attribute
'args'``. ``Policy.use_base()`` snapshots on every ``__enter__``, so it is
safe to re-enter.

Importing this module does not import transformers or peft; only the
``hf_lora`` loader does, when called.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch

from .registry import Registry

LoraState = Dict[str, torch.Tensor]
MODEL_LOADERS: Registry = Registry("model loader")


class _UseBase:
    """Re-enterable context manager: base weights inside, trained weights restored after."""

    def __init__(self, policy: "Policy"):
        self.policy = policy
        self._saved: Optional[LoraState] = None

    def __enter__(self) -> "Policy":
        self._saved = self.policy.snapshot()
        self.policy.restore(self.policy.initial_lora)
        return self.policy

    def __exit__(self, *exc) -> bool:
        if self._saved is not None:
            self.policy.restore(self._saved)
            self._saved = None
        return False


class Policy:
    """Bundles model + tokenizer and owns every adapter-state transition."""

    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = getattr(model, "device", None) or next(model.parameters()).device
        self.initial_lora: LoraState = self.snapshot()
        if not self.initial_lora:
            raise ValueError("model has no LoRA parameters ('lora_' in state_dict keys)")

    # ---------------------------- state machine ---------------------------
    def snapshot(self) -> LoraState:
        return {k: v.detach().clone() for k, v in self.model.state_dict().items() if "lora_" in k}

    def restore(self, snap: LoraState) -> None:
        sd = self.model.state_dict()
        for k, v in snap.items():
            sd[k].copy_(v.to(sd[k].device))

    def reset(self) -> None:
        """Restore the pristine (base-equivalent) adapter. Destroys unsaved training."""
        self.restore(self.initial_lora)

    def adapter_magnitude(self) -> float:
        return float(sum(v.abs().sum() for k, v in self.model.state_dict().items()
                         if "lora_B" in k))

    def is_trained(self, tol: float = 1e-6) -> bool:
        """True iff the LoRA B matrices have moved away from zero. Guards every
        before/after claim: an untrained adapter silently yields identical columns."""
        return self.adapter_magnitude() > tol

    def use_base(self) -> _UseBase:
        return _UseBase(self)

    # ------------------------------ generation ----------------------------
    def stop_token_ids(self) -> List[int]:
        """Every token id that ends a generation: the tokenizer's EOS plus any the
        model's generation config adds (instruct models often have two)."""
        ids = set()
        for src in (getattr(self.tokenizer, "eos_token_id", None),
                    getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)):
            if isinstance(src, int):
                ids.add(src)
            elif src is not None:
                ids.update(int(i) for i in src)
        return sorted(ids)

    # ------------------------------ persistence ---------------------------
    def save_adapter(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({k: v.cpu() for k, v in self.snapshot().items()}, path)
        return path

    def load_adapter(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        self.restore(torch.load(path, map_location="cpu"))
        return True

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]


def load_policy(model_cfg) -> Policy:
    """The policy built by the loader named in ``model_cfg.loader``."""
    return MODEL_LOADERS.get(model_cfg.loader)(model_cfg)


@MODEL_LOADERS.register("hf_lora")
def load_hf_lora(model_cfg) -> Policy:
    """A Hugging Face causal LM with LoRA adapters; 4-bit NF4 base by default (QLoRA)."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name)
    kwargs: dict = {}
    if model_cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_cfg.name, **kwargs)
    model = get_peft_model(model, LoraConfig(
        r=model_cfg.lora_r, lora_alpha=model_cfg.lora_alpha, lora_dropout=model_cfg.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=list(model_cfg.target_modules)))
    model.config.use_cache = False
    # Gradient checkpointing is deliberately OFF: its recompute pass conflicts
    # with adapter toggling ("CheckpointError: 404 tensors saved during forward
    # vs 57 during recomputation"). Compensated by scoring one sequence at a time.
    return Policy(model, tokenizer)


# The CPU stand-in model registers itself as the "tiny" loader.
from . import testing as _testing  # noqa: E402,F401
