"""The documentation must describe the code that exists: commands, configs,
paths, registered names, make targets and test names quoted in the docs are
checked against the implementation."""
import json
import re
from pathlib import Path

import pytest

from grpo.algorithm import ADVANTAGES, LOSSES
from grpo.cli import build_parser
from grpo.config import Config, load_config
from grpo.data import TASKS
from grpo.model import MODEL_LOADERS
from grpo.rewards import REWARD_COMPONENTS

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), ROOT / "CHANGELOG.md",
        ROOT / "notebooks" / "README.md"]
CURRENT = [ROOT / "README.md", ROOT / "docs" / "ARCHITECTURE.md", ROOT / "docs" / "METHOD.md",
           ROOT / "docs" / "TECH_DEBT.md", ROOT / "notebooks" / "README.md"]


def _text(paths):
    return "\n".join(p.read_text() for p in paths)


def _notebook_code():
    nb = json.loads((ROOT / "notebooks" / "quickstart_colab.ipynb").read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_every_grpo_command_in_docs_exists():
    sub = next(a for a in build_parser()._actions if a.dest == "command")
    text = _text(CURRENT) + (ROOT / "Makefile").read_text() + _notebook_code()
    used = set(re.findall(r"(?m)^\s*!?grpo (\w+)", text))           # command lines
    used |= set(re.findall(r"`grpo (\w+)", text))                      # inline code
    assert used and used <= set(sub.choices), used - set(sub.choices)


def test_every_config_named_in_docs_exists_and_loads():
    text = _text(DOCS) + (ROOT / "Makefile").read_text() + _notebook_code()
    names = set(re.findall(r"configs/[\w./*-]+\.yaml", text))
    names = {n for n in names if "*" not in n}
    retired = {"configs/default.yaml", "configs/recommended.yaml"}      # named only as history
    for n in sorted(names - retired):
        load_config(ROOT / n)


def test_every_repository_path_in_current_docs_exists():
    text = _text(CURRENT)
    paths = set(re.findall(r"`((?:src/grpo|grpo|docs|tests|notebooks|configs)/[\w./-]+)`", text))
    missing = []
    for p in paths:
        candidates = [ROOT / p, ROOT / "src" / p]
        if not any(c.exists() for c in candidates):
            missing.append(p)
    assert not missing, missing


def test_registered_names_in_architecture_match_the_registries():
    text = (ROOT / "docs" / "ARCHITECTURE.md").read_text()
    para = text[text.index("Registered today"):].split("\n\n")[0]
    for reg, label in [(TASKS, "tasks"), (REWARD_COMPONENTS, "reward components"),
                       (ADVANTAGES, "advantages"), (LOSSES, "losses"),
                       (MODEL_LOADERS, "model loaders")]:
        seg = para[para.index(label):]
        seg = seg[:min([seg.index(x) for x in (";", ".") if x in seg])]
        named = set(re.findall(r"`([\w]+)`", seg))
        builtin = {n for n in reg.names() if not n.startswith("test_") and n != "brevity"}
        assert named == builtin, (label, named, builtin)


def test_base_yaml_lists_every_setting():
    import yaml
    raw = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    full = Config().to_dict()
    for section, fields in full.items():
        if section == "experiment":
            fields = {k: v for k, v in fields.items() if k != "name"}   # name = file stem
        assert set(fields) <= set(raw[section]), (section, set(fields) - set(raw[section]))


def test_make_targets_used_in_readme_exist():
    targets = set(re.findall(r"^([a-z-]+):", (ROOT / "Makefile").read_text(), re.M))
    readme = (ROOT / "README.md").read_text()
    used = set(re.findall(r"(?m)^\s*make ([a-z-]+)", readme))      # command lines
    used |= set(re.findall(r"`make ([a-z-]+)", readme))              # inline code
    assert {"test", "check", "reproduce", "mutation"} <= used
    assert used <= targets, used - targets


def test_tests_named_in_docs_exist():
    text = _text(DOCS)
    names = set(re.findall(r"`?(test_[a-z0-9_]+)", text))
    source = "\n".join(p.read_text() for p in (ROOT / "tests").glob("test_*.py"))
    defined = set(re.findall(r"def (test_[a-z0-9_]+)", source))
    defined |= {p.stem for p in (ROOT / "tests").glob("test_*.py")}
    assert names <= defined, names - defined


def test_architecture_notebook_snippet_runs(tmp_path, monkeypatch):
    text = (ROOT / "docs" / "ARCHITECTURE.md").read_text()
    snippet = text[text.index("```python\nfrom grpo.config import load_config"):]
    snippet = snippet[len("```python\n"):snippet.index("```", 10)]
    snippet = snippet.replace('"configs/base.yaml"', repr(str(ROOT / "configs" / "cpu_smoke.yaml")))
    snippet = snippet.replace('["experiment.name=brevity_trial"]',
                              f'["experiment.name=brevity_trial", '
                              f'"experiment.output_dir={tmp_path}", "evaluation.probes=false"]')
    if "brevity" in REWARD_COMPONENTS:
        pytest.skip("already registered in this session")
    exec(compile(snippet, "ARCHITECTURE.md", "exec"), {})
    assert (tmp_path / "brevity_trial" / "report.md").exists()
