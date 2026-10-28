from pathlib import Path

import pytest

from grpo.config import Config, ConfigError, fingerprint_of, from_dict, load_config, parse_override

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").rglob("*.yaml"))


def test_defaults_are_valid():
    c = Config()
    assert c.algorithm.group_size >= 2 and c.task.version == 2
    assert [x.name for x in c.reward.components] == ["correct_graded", "format"]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_shipped_config_loads(path):
    cfg = load_config(path)
    assert cfg.experiment.name == path.stem


def test_run6_differs_from_base_only_in_dataset_version():
    base, run6 = load_config(ROOT / "configs/base.yaml").to_dict(), \
        load_config(ROOT / "configs/run6.yaml").to_dict()
    assert run6["task"]["version"] == 1 and base["task"]["version"] == 2
    for d in (base, run6):
        d["task"].pop("version")
        d["experiment"].pop("name")
    assert base == run6


def test_run6_matches_the_reported_run_settings():
    c = load_config(ROOT / "configs/run6.yaml")
    assert (c.model.name, c.model.lora_r, c.model.lora_alpha, c.model.lora_dropout) == \
        ("Qwen/Qwen2.5-0.5B-Instruct", 16, 32, 0.05)
    assert (c.algorithm.group_size, c.algorithm.epsilon, c.algorithm.beta, c.algorithm.mu) == \
        (6, 0.2, 0.04, 1)
    assert (c.sampling.temperature, c.sampling.top_p, c.sampling.max_new_tokens) == (1.0, 0.95, 160)
    assert (c.training.steps, c.training.lr, c.training.schedule_seed) == (60, 1e-5, 1234)
    assert (c.evaluation.n, c.evaluation.k, c.experiment.seed, c.task.seed) == (60, 6, 42, 42)
    comps = [(x.name, x.weight, x.params) for x in c.reward.components]
    assert comps == [("correct_graded", 1.0, {"lam": 0.3, "kappa": 0.2}), ("format", 0.1, {})]


def test_gated_fixes_both_known_issues():
    c = load_config(ROOT / "configs/gated.yaml")
    assert c.reward.components[0].name == "correct_gated" and c.model.lora_dropout == 0.0


def test_extends_merges_sections_and_replaces_lists(tmp_path):
    (tmp_path / "parent.yaml").write_text("algorithm: {mu: 2, beta: 0.1}\n")
    (tmp_path / "child.yaml").write_text(
        "extends: parent.yaml\nalgorithm: {mu: 3}\nreward:\n  components:\n"
        "    - {name: correct_binary}\n")
    c = load_config(tmp_path / "child.yaml")
    assert c.algorithm.mu == 3 and c.algorithm.beta == 0.1
    assert [x.name for x in c.reward.components] == ["correct_binary"]


@pytest.mark.parametrize("text,match", [("algorithm: {group_sise: 6}\n", "group_sise"),
                                        ("trainer: {steps: 3}\n", "trainer")])
def test_unknown_keys_rejected(tmp_path, text, match):
    p = tmp_path / "bad.yaml"
    p.write_text(text)
    with pytest.raises(ConfigError, match=match):
        load_config(p)


@pytest.mark.parametrize("data", [
    {"algorithm": {"group_size": 1}}, {"algorithm": {"mu": 0}}, {"algorithm": {"epsilon": 0.0}},
    {"algorithm": {"beta": -1.0}}, {"task": {"train_pool": 600}}, {"evaluation": {"n": 500}},
    {"sampling": {"temperature": 0.0}}, {"training": {"steps": 0}},
    {"algorithm": {"group_size": "six"}}, {"reward": {"components": []}},
])
def test_invalid_values_rejected(data):
    with pytest.raises(ConfigError):
        from_dict(data)


@pytest.mark.parametrize("key,value", [("task.name", "nope"), ("algorithm.loss", "nope"),
                                       ("algorithm.advantage", "nope"), ("model.loader", "nope")])
def test_unregistered_names_rejected(key, value):
    with pytest.raises(ConfigError, match="unknown"):
        load_config(None, [f"{key}={value}"])


def test_unregistered_reward_component_rejected():
    with pytest.raises(ConfigError, match="reward component"):
        load_config(None, {"reward.components": [{"name": "nope"}]})


def test_overrides_are_typed():
    assert parse_override("algorithm.mu=4") == ("algorithm.mu", 4)
    c = load_config(None, ["algorithm.mu=4", "training.lr=1e-5", "evaluation.probes=false"])
    assert c.algorithm.mu == 4 and c.training.lr == 1e-5 and c.evaluation.probes is False


def test_with_overrides_is_an_immutable_copy():
    c = Config()
    d = c.with_overrides({"algorithm.beta": 0.0})
    assert d.algorithm.beta == 0.0 and c.algorithm.beta == 0.04


def test_round_trip_through_dict():
    c = load_config(ROOT / "configs/gated.yaml")
    assert from_dict(c.to_dict()) == c


def test_fingerprint_tracks_only_selected_settings():
    c = Config()
    fp = fingerprint_of(c, ["model.name", "task"])
    assert fingerprint_of(c.with_overrides({"algorithm.mu": 4}), ["model.name", "task"]) == fp
    assert fingerprint_of(c.with_overrides({"task.version": 1}), ["model.name", "task"]) != fp
