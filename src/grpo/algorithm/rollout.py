"""Model-facing primitives: prompts in, sampled answers and log-probabilities out."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable, List, Sequence, Tuple, Union

import torch


def build_prompt_ids(policy, messages: List[dict]) -> torch.Tensor:
    """Chat-template the messages and return input ids on the policy's device."""
    text = policy.tokenizer.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True)
    return policy.tokenizer(text, return_tensors="pt").input_ids.to(policy.device)


def generate_group(policy, prompt_ids: torch.Tensor, group_size: int,
                   sampling) -> Tuple[torch.Tensor, int]:
    """Sample the whole group in ONE batched call (generation is most of the
    wall-clock). Sampling is the only source of variety within a group."""
    policy.model.eval()
    with torch.no_grad():
        out = policy.model.generate(
            prompt_ids, max_new_tokens=sampling.max_new_tokens, do_sample=True,
            temperature=sampling.temperature, top_p=sampling.top_p,
            num_return_sequences=group_size, pad_token_id=policy.tokenizer.pad_token_id)
    return out, prompt_ids.shape[1]


def true_lengths(seqs: torch.Tensor, plen: int,
                 stop_ids: Union[int, Iterable[int]]) -> List[int]:
    """Full length of each sequence: up to and INCLUDING its first stop token
    after the prompt (stopping is a behaviour to be scored), or the whole row if
    it never stopped. Accepts every stop id the model uses: a sequence that ends
    on a second stop token (which may equal the pad token) must not be treated
    as running into its padding."""
    stop = torch.tensor(sorted({stop_ids} if isinstance(stop_ids, int) else set(stop_ids)),
                        device=seqs.device)
    lengths = []
    for row in seqs:
        hit = torch.isin(row[plen:], stop).nonzero()
        lengths.append(plen + (hit[0].item() + 1 if len(hit) else len(row) - plen))
    return lengths


def completion_mask(G: int, T: int, plen: int, lengths: Sequence[int],
                    device=None) -> torch.Tensor:
    """mask[i, plen-1 : L_i-1] = 1.

    Output position t scores token t+1, so answer tokens x[plen..L-1] are
    scored at positions plen-1..L-2. Starting at ``plen`` (the natural-looking
    choice) silently drops the FIRST generated token from every gradient.
    """
    m = torch.zeros(G, T - 1, device=device)
    for i, L in enumerate(lengths):
        m[i, plen - 1:L - 1] = 1.0
    return m


def per_token_logps(policy, seqs: torch.Tensor, use_adapter: bool = True, chunk: int = 1,
                    requires_grad: bool = False) -> torch.Tensor:
    """log p(x[t+1] | x[<=t]) at every position, shape [B, T-1].

    Chunked because the memory hog is the logits, not the weights: [G, T,
    152k vocab] in fp32 would not fit a T4. ``use_adapter=False`` scores with
    the REFERENCE policy via ``policy.use_base()`` (explicit tensor swap,
    re-enterable).
    """
    grad_ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    state_ctx = nullcontext() if use_adapter else policy.use_base()
    pad_id = policy.tokenizer.pad_token_id if policy.tokenizer is not None else -1
    outs = []
    with state_ctx:
        for i in range(0, seqs.shape[0], chunk):
            b = seqs[i:i + chunk]
            attn = (b != pad_id).long()
            with grad_ctx:
                logits = policy.model(b, attention_mask=attn).logits[:, :-1]
                lp = torch.log_softmax(logits.float(), dim=-1)
                outs.append(lp.gather(2, b[:, 1:].unsqueeze(-1)).squeeze(-1))
    return torch.cat(outs, 0)
