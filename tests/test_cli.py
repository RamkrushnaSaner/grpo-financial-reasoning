"""The real CLI end to end with the tiny CPU model standing in for Qwen
(configs/cpu_smoke.yaml). Catches wiring bugs unit tests miss: argument
parsing, run directories, fingerprinted caching, report and plot writing."""
import json
from pathlib import Path

import pytest
import yaml

from grpo.cli import build_parser, main

ROOT = Path(__file__).resolve().parents[1]
SMOKE = str(ROOT / "configs" / "cpu_smoke.yaml")
ARTIFACTS = ("config.yaml", "manifest.json", "run.log", "metrics.jsonl", "train.json",
             "adapter.pt", "eval_after.json", "comparison.json", "probes.json", "report.md",
             "curves.png", "before_after.png")


def _run(tmp_path, *extra):
    out = tmp_path / "results"
    assert main(["run", SMOKE, "--set", f"experiment.output_dir={out}", *extra]) == 0
    return out


def test_validate(capsys):
    assert main(["validate"]) == 0
    assert "ALL PASS" in capsys.readouterr().out


def test_config_prints_the_resolved_config(capsys):
    main(["config", SMOKE, "--set", "algorithm.mu=2"])
    cfg = yaml.safe_load(capsys.readouterr().out)
    assert cfg["algorithm"]["mu"] == 2 and cfg["model"]["loader"] == "tiny"
    assert cfg["experiment"]["name"] == "cpu_smoke"


def test_run_writes_every_artifact(tmp_path):
    out = _run(tmp_path)
    run = out / "cpu_smoke"
    for f in ARTIFACTS:
        assert (run / f).exists(), f
    assert len(list((out / "_baselines").glob("*.json"))) == 1
    report = (run / "report.md").read_text()
    assert "gradient-direction correlation" in report and "PROBE SUMMARY" in report
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["fingerprints"]["training"] == json.loads(
        (run / "train.json").read_text())["fingerprint"]
    after = json.loads((run / "eval_after.json").read_text())
    assert len(after["records"]) == after["n"]


def test_rerun_resumes_from_caches_and_a_change_invalidates_them(tmp_path):
    out = _run(tmp_path)
    run = out / "cpu_smoke"
    mtimes = {f: (run / f).stat().st_mtime for f in ("adapter.pt", "eval_after.json")}
    baseline = next((out / "_baselines").glob("*.json"))
    base_mtime = baseline.stat().st_mtime
    _run(tmp_path, "--no-probes")                                  # nothing changed
    assert {f: (run / f).stat().st_mtime for f in mtimes} == mtimes
    assert baseline.stat().st_mtime == base_mtime
    assert "cache hit" in (run / "run.log").read_text()
    _run(tmp_path, "--no-probes", "--set", "training.steps=3")      # training changed
    assert (run / "adapter.pt").stat().st_mtime != mtimes["adapter.pt"]
    assert baseline.stat().st_mtime == base_mtime                 # baseline still valid
    assert len(json.loads((run / "train.json").read_text())["history"]["kl"]) == 3


def test_ask_after_run(tmp_path, capsys):
    out = _run(tmp_path, "--no-probes")
    capsys.readouterr()
    main(["ask", SMOKE, "--set", f"experiment.output_dir={out}", "--question", "What was revenue?",
          "--context", "Revenue was $5.0B.", "--answer", "5.0", "--samples", "2"])
    assert "TRAINED" in capsys.readouterr().out


def test_ask_without_a_trained_run_fails_clearly(tmp_path):
    with pytest.raises(SystemExit, match="no trained run"):
        main(["ask", SMOKE, "--set", f"experiment.output_dir={tmp_path}", "--question", "q"])


def test_sweep_and_summarize(tmp_path, capsys):
    out = tmp_path / "results"
    other = tmp_path / "binary.yaml"
    other.write_text(f"extends: {SMOKE}\nreward:\n  components:\n"
                     "    - {name: correct_binary}\n    - {name: format, weight: 0.1}\n")
    main(["sweep", SMOKE, str(other), "--set", f"experiment.output_dir={out}", "--no-probes"])
    text = capsys.readouterr().out
    assert "cpu_smoke" in text and "binary" in text
    main(["summarize", str(out / "cpu_smoke"), str(out / "binary"), str(out / "missing")])
    assert "(incomplete)" in capsys.readouterr().out


def test_parser_lists_the_documented_commands():
    sub = next(a for a in build_parser()._actions if a.dest == "command")
    assert set(sub.choices) == {"validate", "config", "run", "sweep", "summarize", "ask"}


def test_logger_switches_file_between_runs(tmp_path):
    """Regression: get_logger used to return early once ANY handler existed, so a
    second run in the same session never wrote its own log file."""
    from grpo.tracking import get_logger
    a, b = tmp_path / "a" / "run.log", tmp_path / "b" / "run.log"
    get_logger(log_file=a).info("first")
    get_logger(log_file=b).info("second")
    assert "first" in a.read_text() and "second" not in a.read_text()
    assert "second" in b.read_text()
