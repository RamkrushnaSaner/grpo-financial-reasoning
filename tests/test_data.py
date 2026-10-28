import hashlib
import json
import re

import pytest

from grpo.config import TaskConfig
from grpo.data import TASKS, FinancialTask, build_task, generate, verify_answer

# sha256 of the 500-question dataset produced by the ORIGINAL generator (seed 42),
# recorded before the refactor. Version 1 must reproduce it exactly.
LEGACY_SHA256 = "2819fdd7209f1e04f34c1e59c82ad351a0e3f565d6bec2aad480f45c41b86435"


def _implied(s):
    """The answer computed from the numbers the question actually shows."""
    c = s["context"]
    if s["qtype"] == "growth":
        new, old = [float(x) for x in re.findall(r"\$(-?\d+(?:\.\d+)?)B", c)]
        return (new - old) / old * 100
    if s["qtype"] == "margin":
        rev, profit = [float(x) for x in re.findall(r"\$(-?\d+(?:\.\d+)?)B", c)]
        return profit / rev * 100
    if s["qtype"] == "projection":
        cur = float(re.findall(r"\$(\d+(?:\.\d+)?)B", c)[0])
        return cur * (1 + float(re.findall(r"(\d+)% annual", c)[0]) / 100)
    return sum(float(x) for x in re.findall(r"=\$(\d+(?:\.\d+)?)B", c))


def _inconsistent(samples):
    return [i for i, s in enumerate(samples)
            if not verify_answer(_implied(s), s["answer"], s["answer_type"])]


def test_version1_reproduces_the_original_dataset_exactly():
    blob = json.dumps(generate(500, 42, version=1), sort_keys=True)
    assert hashlib.sha256(blob.encode()).hexdigest() == LEGACY_SHA256


def test_version1_label_bug_is_documented_precisely():
    v1 = generate(500, 42, version=1)
    bad = _inconsistent(v1)
    assert all(v1[i]["qtype"] == "growth" for i in bad)
    assert len([i for i in bad if i < 400]) == 63
    assert len([i for i in bad if 400 <= i < 460]) == 11
    assert v1[412]["answer"] == 17.0 and "$1.2B, compared to $1.0B" in v1[412]["context"]


def test_version2_asks_the_same_questions_with_consistent_labels():
    v1, v2 = generate(500, 42, version=1), generate(500, 42, version=2)
    assert [(s["context"], s["question"], s["qtype"]) for s in v1] == \
        [(s["context"], s["question"], s["qtype"]) for s in v2]
    assert _inconsistent(v2) == []
    assert v2[412]["answer"] == 20.0


def test_dataset_is_deterministic_and_covers_all_archetypes():
    a = FinancialTask(size=500, seed=42)
    assert a.samples == FinancialTask(size=500, seed=42).samples
    assert {s["qtype"] for s in a.samples} == {"growth", "margin", "comparison", "projection"}
    assert {a.subgroup(s) for s in a.samples} == {"1step", "2step"}


def test_unknown_version_rejected():
    with pytest.raises(ValueError):
        generate(5, 42, version=3)


def test_verify_answer_tolerance_and_safety():
    assert verify_answer(21.3, 21.1, "percentage") == 1.0
    assert verify_answer(21.7, 21.1, "percentage") == 0.0
    assert verify_answer(8.45, 8.4, "dollar") == 1.0
    assert verify_answer(80.4, 80, "integer") == 1.0
    for bad in [None, "abc", float("nan"), float("inf")]:
        assert verify_answer(bad, 5.0, "dollar") == 0.0
    assert verify_answer(5.0, None, "dollar") == 0.0


def test_tolerance_override_changes_verification():
    loose = FinancialTask(size=10, train_pool=8)
    tight = FinancialTask(size=10, train_pool=8, tolerance={"dollar": 0.01})
    s = {"answer": 5.09, "answer_type": "dollar"}
    assert loose.verify(5.184, s) and not tight.verify(5.184, s)     # the A4 probe case


def test_schedule_matches_the_original_and_stays_in_the_pool():
    t = FinancialTask(size=500, train_pool=400)
    s = t.schedule(60, 1234)
    assert s[:5] == [303, 211, 294, 53, 204] and s == t.schedule(60, 1234)
    assert all(0 <= i < 400 for i in s) and len(set(s)) == 56


def test_eval_indices_are_the_first_held_out_questions():
    t = FinancialTask(size=500, train_pool=400)
    assert t.eval_indices(60) == list(range(400, 460))
    with pytest.raises(ValueError):
        t.eval_indices(101)


def test_task_registry_and_builder():
    assert "financial" in TASKS
    t = build_task(TaskConfig(version=1, size=50, train_pool=40))
    assert isinstance(t, FinancialTask) and len(t) == 50 and t.version == 1


def test_probes_are_well_formed():
    probes = FinancialTask(size=10, train_pool=8).probes()
    assert len(probes) == 15 and len({p["id"] for p in probes}) == 15
    assert all("question" in p and "expect" in p for p in probes)


def test_parse_strict_and_fallback():
    t = FinancialTask(size=10, train_pool=8)
    assert t.parse("Final answer: 8.7") == 8.7
    assert t.parse("Final answer: 8.7", strict=True) is None
    assert t.follows_format("Answer: 3") and not t.follows_format("answer: 3")
