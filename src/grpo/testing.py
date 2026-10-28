"""A tiny CPU stand-in for the real model, used by the test suite and by
``configs/cpu_smoke.yaml``.

It exercises the REAL training, evaluation and probe code paths end to end
in seconds, with no download and no GPU. Its numbers mean nothing: it checks
that the pipeline runs and that every artifact is written, not that anything
is learned.

    vocabulary: 0 pad, 1 eos, 2 "Answer: ", 3-12 digits, 13 " ", 14 ".", 15-19 letters
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

VOCAB = {0: "", 1: "", 2: "Answer: ", 13: " ", 14: "."}
VOCAB.update({3 + d: str(d) for d in range(10)})
VOCAB.update({15 + i: c for i, c in enumerate("abcde")})
V = 20


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(m["content"] for m in messages)

    def __call__(self, text, return_tensors="pt"):
        ids = [(ord(c) % 17) + 2 for c in text][:12]
        return SimpleNamespace(input_ids=torch.tensor([ids]))

    def decode(self, ids, skip_special_tokens=True):
        return "".join(VOCAB[int(i)] for i in ids if int(i) not in (0, 1))


class TinyLoRALM(nn.Module):
    """Frozen 'base' layers + trainable LoRA A/B. lora_B starts at zero, so the
    pristine adapter IS the base model, exactly as with peft."""

    def __init__(self, dim: int = 16, r: int = 4, seed: int = 0, lora_dropout: float = 0.0):
        super().__init__()
        self.lora_dropout = nn.Dropout(lora_dropout)
        g = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(V, dim)
        self.base = nn.Linear(dim, V)
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn(V, dim, generator=g))
            self.base.weight.copy_(torch.randn(V, dim, generator=g) * 0.3)
        self.lora_A = nn.Linear(dim, r, bias=False)
        self.lora_B = nn.Linear(r, V, bias=False)
        nn.init.zeros_(self.lora_B.weight)
        for p in list(self.emb.parameters()) + list(self.base.parameters()):
            p.requires_grad_(False)

    def forward(self, x, attention_mask=None):
        h = self.emb(x)
        return SimpleNamespace(
            logits=self.base(h) + self.lora_B(self.lora_A(self.lora_dropout(h))))

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=8, do_sample=False, temperature=1.0,
                 top_p=1.0, num_return_sequences=1, pad_token_id=0):
        seqs = input_ids.repeat(num_return_sequences, 1)
        done = torch.zeros(seqs.shape[0], dtype=torch.bool)
        for _ in range(max_new_tokens):
            logits = self.forward(seqs).logits[:, -1]
            if do_sample:
                nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1).squeeze(1)
            else:
                nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, pad_token_id), nxt)
            done = done | (nxt == 1)
            seqs = torch.cat([seqs, nxt.unsqueeze(1)], dim=1)
            if done.all():
                break
        return seqs


def tiny_policy(lora_dropout: float = 0.0, seed: int = 0):
    from .model import Policy
    torch.manual_seed(seed)
    return Policy(TinyLoRALM(lora_dropout=lora_dropout), FakeTokenizer())


def _register() -> None:
    from .model import MODEL_LOADERS

    @MODEL_LOADERS.register("tiny")
    def load_tiny(model_cfg):
        """CPU stand-in; ignores ``name`` and quantisation, honours ``lora_dropout``."""
        return tiny_policy(lora_dropout=model_cfg.lora_dropout)


_register()
