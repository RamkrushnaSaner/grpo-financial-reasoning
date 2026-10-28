import numpy as np
import pytest

from grpo.algorithm import group_advantages
from grpo.config import RewardComponentConfig, RewardConfig
from grpo.data import FinancialTask, extract_answer, verify_answer
from grpo.rewards import (
    REWARD_COMPONENTS,
    CompositeReward,
    RewardComponent,
    build_reward,
    graded_score,
    partial_mass,
    validate_reward,
)

TASK = FinancialTask(size=10, train_pool=8)
TRUTH = {"answer": 39.0, "answer_type": "percentage"}
GROUP = ["Margin = 4.6/11.8 = 0.39\nAnswer: 39", "Answer: 38.98", "Answer: 37.33",
         "Answer: 256.5", "roughly 30 percent", "I cannot determine this."]


def reward(*comps):
    return build_reward(RewardConfig(tuple(RewardComponentConfig(*c) for c in comps)), TASK)


GRADED = (("correct_graded", 1.0, {"lam": 0.3, "kappa": 0.2}), ("format", 0.1, {}))
BINARY = (("correct_binary", 1.0, {}), ("format", 0.1, {}))
GATED = (("correct_gated", 1.0, {"lam": 0.3, "kappa": 0.2}), ("format", 0.1, {}))


@pytest.mark.parametrize("text,expected", [
    ("Answer: 23%", 23.0), ("Answer: 8.4 B", 8.4), ("Answer: 5\nAnswer: 7.5", 7.5),
    ("Answer: $2.24", 2.24), ("Final Answer: 37%", 37.0), ("Answer: -12.5", -12.5),
    ("no numbers here", None),
])
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


def test_malformed_number_does_not_crash():
    assert extract_answer("Answer: 8.25.0.0") is not None


def test_fallback_can_be_disabled():
    assert extract_answer("Final answer: 8.7") == 8.7
    assert extract_answer("Final answer: 8.7", allow_fallback=False) is None


def test_validation_suite_passes_and_rejects_unsafe_lambda():
    validate_reward(0.3, 0.2)
    with pytest.raises(AssertionError, match="SAFETY"):
        validate_reward(0.6, 0.2)


def test_graded_values():
    g = lambda p: graded_score(p, TRUTH, TASK.verify, 0.3, 0.2)
    assert g(39.0) == 1.0
    assert g(38.7) == 1.0                                         # within tolerance
    assert abs(g(37.33) - 0.2422) < 1e-3
    assert g(256.5) < 1e-6
    assert g(float("nan")) == 0.0 and g(None) == 0.0


def test_near_miss_is_distinguished_only_by_graded():
    b, _ = reward(*BINARY).score_group(GROUP, TRUTH)
    g, _ = reward(*GRADED).score_group(GROUP, TRUTH)
    assert b[2] == b[3] == 0.1                                    # binary: near miss == blunder
    assert g[2] > g[3]                                            # graded ranks them
    assert len({round(x, 6) for x in b}) == 3 and len({round(x, 6) for x in g}) == 5


def test_gated_component_closes_the_format_loophole():
    """Run 6's failure: 'Final answer: 8.7' earned full correctness credit through
    the parser fallback while skipping the format reward."""
    s = {"answer": 8.7, "answer_type": "dollar"}
    graded, gated = reward(*GRADED), reward(*GATED)
    loophole, contract = "Total = 8.7\nFinal answer: 8.7", "Total = 8.7\nAnswer: 8.7"
    assert graded.score(loophole, s)[0] == pytest.approx(1.0)
    assert gated.score(loophole, s)[0] == 0.0
    assert graded.score(contract, s)[0] == pytest.approx(1.1)
    assert gated.score(contract, s)[0] == pytest.approx(1.1)


def test_gated_rejects_every_contract_violation():
    s = {"answer": 8.7, "answer_type": "dollar"}
    gated = reward(*GATED)
    for text in ["8.7", "Final answer: 8.7", "Answer: eight point seven", "answer: 8.7"]:
        assert gated.score(text, s)[1]["correct_gated"] == 0.0, text
    assert gated.score("Answer: 8.7", s)[1]["correct_gated"] == 1.0


def test_binary_reproduces_the_original_reward():
    reg = reward(*BINARY)
    for text, gt, at in [("Answer: 39", 39.0, "percentage"), ("Answer: 37.33", 39.0, "percentage"),
                         ("about 30 percent", 39.0, "percentage"), ("Answer: 8.4 B", 8.4, "dollar"),
                         ("cannot determine", 5.1, "dollar")]:
        original = verify_answer(extract_answer(text), gt, at) + 0.1 * ("Answer:" in text)
        assert reg.score(text, {"answer": gt, "answer_type": at})[0] == pytest.approx(original)


def test_fault_isolation():
    reg = CompositeReward([RewardComponent("boom", lambda t, s: 1 / 0, 1.0),
                           RewardComponent("nan", lambda t, s: float("nan"), 1.0),
                           RewardComponent("format", lambda t, s: 1.0, 0.1)])
    total, _ = reg.score("Answer: 5", TRUTH)
    assert total == pytest.approx(0.1)
    assert reg.failures == {"boom": 1, "nan": 1, "format": 0}


def test_unknown_component_and_bad_params_rejected():
    with pytest.raises(KeyError, match="reward component"):
        reward(("nope", 1.0, {}))
    with pytest.raises(ValueError, match="bad params"):
        reward(("format", 0.1, {"lam": 0.3}))


def test_lambda_violation_warns_at_build_time():
    with pytest.warns(UserWarning, match="2\\*lam < 1"):
        reward(("correct_graded", 1.0, {"lam": 0.6, "kappa": 0.2}))


def test_custom_component_plugs_in_by_name():
    @REWARD_COMPONENTS.register("test_brevity")
    def brevity(task, max_chars=10):
        return lambda t, s: float(len(t) <= max_chars)
    r = reward(("test_brevity", 0.5, {"max_chars": 5}), ("format", 0.1, {}))
    assert r.score("Answer: 1", TRUTH)[1] == {"test_brevity": 0.0, "format": 1.0}
    assert r.score("abc", TRUTH)[0] == pytest.approx(0.5)


def test_affine_invariance_and_saturated_component_is_inert():
    r = np.array([1.1, 0.1, 1.1, 0.1, 0.1, 0.0])
    base = group_advantages(r.tolist())
    for transformed in (r * 100, r + 7):
        assert np.allclose(group_advantages(transformed.tolist()).numpy(), base.numpy(), atol=1e-3)
    corr = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert np.allclose(group_advantages(corr).numpy(),
                       group_advantages([c + 0.1 for c in corr]).numpy(), atol=1e-6)


def test_partial_mass_ignores_correct_answers_and_uses_fixed_lambda():
    texts = ["Answer: 39", "Answer: 37.33", "Answer: 256.5"]
    pm = partial_mass(TASK, texts, TRUTH, 0.3, 0.2)
    assert pm == pytest.approx((0.24222 + 0.0) / 2, abs=1e-3)
    assert partial_mass(TASK, texts, TRUTH, 0.6, 0.2) == pytest.approx(2 * pm)
