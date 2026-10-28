"""The REAL Hugging Face + peft code path (``model.loader: hf_lora``) on a tiny,
randomly initialised Qwen2 model built locally: no download, no GPU.

What it covers that the tiny stand-in cannot: AutoTokenizer chat templates,
AutoModelForCausalLM loading, peft LoRA injection and state-dict keys, HF
``generate`` with groups, two stop tokens where one equals the pad token, and
adapter swapping for the reference policy. What it cannot cover: 4-bit
quantisation (bitsandbytes needs a GPU) and real Qwen weights.

Skipped when transformers/peft are not installed (``pip install -e ".[train]"``).
"""
import json

import pytest

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")
tokenizers = pytest.importorskip("tokenizers")

from conftest import tiny_config  # noqa: E402

from grpo.experiment import run_pipeline  # noqa: E402
from grpo.model import load_policy  # noqa: E402

CHAT_TEMPLATE = ("{% for m in messages %}{{'<|im_start|>' + m['role'] + '\n' + m['content'] + "
                 "'<|im_end|>\n'}}{% endfor %}{% if add_generation_prompt %}"
                 "{{'<|im_start|>assistant\n'}}{% endif %}")
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


@pytest.fixture(scope="module")
def tiny_qwen(tmp_path_factory):
    """A 2-layer Qwen2 with a byte-level BPE tokenizer, Qwen-style special tokens
    and a generation config that stops on BOTH <|im_end|> and <|endoftext|>,
    where <|endoftext|> is also the pad token: the harder case for stop handling
    (check the real model's ``generation_config.eos_token_id`` on a GPU run)."""
    from transformers import (
        GenerationConfig,
        PreTrainedTokenizerFast,
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    d = tmp_path_factory.mktemp("tiny_qwen")
    bpe = tokenizers.ByteLevelBPETokenizer()
    corpus = ["Revenue was $5.0B and profit was $1.2B. Answer: 23.5 Growth = (2.0 - 1.6) / 1.6",
              "What was the operating margin in percent? Final answer: 8.7 total quarters"] * 20
    bpe.train_from_iterator(corpus, vocab_size=400, special_tokens=SPECIAL)
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<|im_end|>",
                                  pad_token="<|endoftext|>")
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(d)
    eos, pad = tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        tie_word_embeddings=True, eos_token_id=eos, pad_token_id=pad))
    model.generation_config = GenerationConfig(eos_token_id=[eos, pad], pad_token_id=pad)
    model.save_pretrained(d)
    return str(d), eos, pad


def _cfg(tmp_path, name):
    return tiny_config(tmp_path,
                       model=dict(loader="hf_lora", name=name, load_in_4bit=False, lora_r=4,
                                  lora_alpha=8, target_modules=["q_proj", "v_proj"]),
                       sampling=dict(max_new_tokens=8), training=dict(steps=2, window=1),
                       evaluation=dict(n=2, k=2, probes=True))


def test_hf_lora_policy_has_lora_state_and_both_stop_tokens(tiny_qwen, tmp_path):
    name, eos, pad = tiny_qwen
    policy = load_policy(_cfg(tmp_path, name).model)
    assert policy.initial_lora and not policy.is_trained()
    assert all("lora_" in k for k in policy.initial_lora)
    assert policy.stop_token_ids() == sorted({eos, pad})


def test_full_pipeline_on_the_hf_code_path(tiny_qwen, tmp_path):
    name, _, _ = tiny_qwen
    out = run_pipeline(_cfg(tmp_path, name))
    run_dir = tmp_path / "results" / "t"
    assert out["policy"].is_trained()
    for f in ("adapter.pt", "train.json", "eval_after.json", "comparison.json", "probes.json",
              "report.md"):
        assert (run_dir / f).exists(), f
    train = json.loads((run_dir / "train.json").read_text())
    assert len(train["history"]["kl"]) == 2
    assert all(len(t) >= 0 for t in train["samples"].values())
