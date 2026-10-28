"""Before/after comparison with paired statistics, plots, and the run report.

Both models answer the SAME held-out questions, so every comparison is paired:
yes/no outcomes (correct, format) use the exact McNemar test on the questions
whose outcome changed; per-question numbers (pass@k, length) use a paired
bootstrap interval. Results saved without per-question records (older runs)
are reported as deltas only, marked "untested".
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..stats import paired_binary, paired_bootstrap_ci, standard_error_pp

ALPHA = 0.05


def _by_index(ev: dict) -> Dict[int, dict]:
    return {r["index"]: r for r in ev.get("records", [])}


def compare(before: dict, after: dict) -> dict:
    """Rows of (metric, before, after, delta, test, p or CI, verdict)."""
    rb, ra = _by_index(before), _by_index(after)
    common = sorted(set(rb) & set(ra))
    paired = bool(common)
    k = after.get("k") or before.get("k") or 0
    groups = sorted(key[4:] for key in after if key.startswith("acc_") and key != "acc_overall")
    rows: List[dict] = []

    def binary_row(label, key, field, subset=None):
        b, a = before.get(key), after.get(key)
        if b is None or a is None:
            return
        row = dict(label=label, key=key, before=b, after=a, delta_pp=(a - b) * 100,
                   n=after.get("n") if subset is None else after.get(f"n_{subset}"))
        if paired:
            idx = [i for i in common if subset is None or rb[i]["subgroup"] == subset]
            t = paired_binary([rb[i][field] for i in idx], [ra[i][field] for i in idx])
            row.update(test="McNemar", p_value=t["p_value"], worse=t["worse"], better=t["better"],
                       verdict="significant" if t["p_value"] < ALPHA else "not significant")
        else:
            row.update(test="none", verdict="untested (no per-question records)")
        rows.append(row)

    def continuous_row(label, key, field, scale=100.0):
        b, a = before.get(key), after.get(key)
        if b is None or a is None:
            return
        row = dict(label=label, key=key, before=b, after=a, delta_pp=(a - b) * scale,
                   n=after.get("n"), scale=scale)
        if paired and all(field in rb[i] and field in ra[i] for i in common):
            ci = paired_bootstrap_ci([rb[i][field] for i in common], [ra[i][field] for i in common])
            excludes_zero = ci["ci_low"] > 0 or ci["ci_high"] < 0
            row.update(test="paired bootstrap", ci_low=ci["ci_low"] * scale,
                       ci_high=ci["ci_high"] * scale,
                       verdict="significant" if excludes_zero else "not significant")
        else:
            row.update(test="none", verdict="untested (no per-question records)")
        rows.append(row)

    binary_row("Overall accuracy", "acc_overall", "correct")
    for g in groups:
        binary_row(f"{g} accuracy", f"acc_{g}", "correct", subset=g)
    binary_row("Format compliance", "format", "format")
    if k:
        continuous_row("pass@1 (sampled)", "pass@1", "pass_at_1")
        continuous_row(f"pass@{k} (sampled)", f"pass@{k}", "pass_at_k")
    continuous_row("Mean length (tokens)", "mean_len", "tokens", scale=1.0)

    acc = rows[0] if rows else None
    if acc is None:
        verdict = "No accuracy to compare."
    elif acc["test"] == "none":
        verdict = (f"Accuracy changed {acc['delta_pp']:+.1f} pp; significance not computed "
                   "(per-question records missing).")
    elif acc["p_value"] < ALPHA:
        direction = "improved" if acc["delta_pp"] > 0 else "reduced"
        verdict = (f"GRPO {direction} accuracy by {acc['delta_pp']:+.1f} pp "
                   f"(McNemar p = {acc['p_value']:.2g}; {acc['better']} questions fixed, "
                   f"{acc['worse']} broken). Single seed.")
    else:
        verdict = (f"No significant accuracy change ({acc['delta_pp']:+.1f} pp, "
                   f"McNemar p = {acc['p_value']:.2g}).")
    gap = None
    if k and "pass@1" in before and f"pass@{k}" in before and "pass@1" in after:
        gap = (before[f"pass@{k}"] - before["pass@1"], after[f"pass@{k}"] - after["pass@1"])
    return dict(n=after.get("n"), k=k, paired=paired, rows=rows, gap=gap, verdict=verdict)


def format_comparison(comp: dict, W: int = 92) -> str:
    n = comp["n"] or 0
    L = ["=" * W, " BEFORE vs AFTER -- same questions, prompt and greedy decoding", "=" * W,
         f" n={n} held out | paired tests: {'yes' if comp['paired'] else 'NO (records missing)'}"
         f" | rough precision of one proportion: +/-{standard_error_pp(max(n, 1)):.1f} pp", "",
         f" {'metric':<24}{'BEFORE':>9}{'AFTER':>9}{'CHANGE':>11}{'n':>5}  test / verdict",
         " " + "-" * (W - 2)]
    for r in comp["rows"]:
        if r["key"] == "mean_len":
            b, a, d = f"{r['before']:.0f}", f"{r['after']:.0f}", f"{r['delta_pp']:+.1f}"
        else:
            b, a, d = f"{r['before']:.1%}", f"{r['after']:.1%}", f"{r['delta_pp']:+.1f} pp"
        if r["test"] == "McNemar":
            t = f"McNemar p={r['p_value']:.2g} (+{r['better']}/-{r['worse']}) {r['verdict']}"
        elif r["test"] == "paired bootstrap":
            t = f"95% CI [{r['ci_low']:+.1f}, {r['ci_high']:+.1f}] {r['verdict']}"
        else:
            t = r["verdict"]
        L.append(f" {r['label']:<24}{b:>9}{a:>9}{d:>11}{str(r['n']):>5}  {t}")
    if comp["gap"]:
        gb, ga = comp["gap"]
        L += ["", f" sampling gap pass@{comp['k']} - pass@1: {gb:.1%} -> {ga:.1%} "
                  f"({(ga - gb) * 100:+.1f} pp). A narrowing gap is the signature of RL "
                  "concentrating existing capability; interpret only if it exceeds the noise."]
    L += ["", " VERDICT: " + comp["verdict"]]
    fmt = next((r for r in comp["rows"] if r["key"] == "format"), None)
    if fmt and fmt["delta_pp"] < 0 and fmt.get("verdict") == "significant":
        L.append(f" WARNING: format compliance fell {fmt['delta_pp']:+.1f} pp. A parser fallback "
                 "may be letting correctness fire without the output contract; see the "
                 "correct_gated reward component.")
    L.append(" SCOPE: in-distribution held-out questions, one seed. Not a general-reasoning claim.")
    return "\n".join(L)


# ------------------------------------------------------------------ plots
def plot_training(history: dict, path: str, group_size: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = max(3, len(history["exact_rate"]) // 10)

    def smooth(x):
        v = np.array([np.nan if e is None else float(e) for e in x])
        v = np.where(np.isnan(v), np.nanmean(v) if np.isfinite(np.nanmean(v)) else 0.0, v)
        return np.convolve(v, np.ones(w) / w, mode="valid")

    panels = [("exact_rate", "Exact correctness (sampled)", "#c0392b"),
              ("pass_at_g", f"pass@{group_size} (sampled)", "#16a085"),
              ("format_rate", "Format compliance", "#2e9e5b"),
              ("skipped", "Degenerate-group rate", "#7f8c8d"),
              ("kl", "KL from reference", "#8e44ad")]
    fig, axes = plt.subplots(1, 5, figsize=(23, 3.8))
    for ax, (key, title, color) in zip(axes, panels, strict=True):
        ax.plot(smooth(history[key]), color=color, lw=2)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(f"GRPO step ({w}-step moving average)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_before_after(before: dict, after: dict, path: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted(k for k in after if k.startswith("acc_") and k != "acc_overall")
    keys = groups + ["acc_overall", "format"]
    labels = [k[4:] if k.startswith("acc_") else k for k in keys]
    bv = [before.get(k, 0) for k in keys]
    av = [after.get(k, 0) for k in keys]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.18, bv, 0.36, label="BEFORE (base)", color="#95a5a6")
    ax.bar(x + 0.18, av, 0.36, label="AFTER (GRPO)", color="#2c6fbb")
    for i, (b, a) in enumerate(zip(bv, av, strict=True)):
        ax.text(i - 0.18, b + 0.012, f"{b:.0%}", ha="center", fontweight="bold")
        ax.text(i + 0.18, a + 0.012, f"{a:.0%}", ha="center", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(max(bv + av) * 1.25, 0.1))
    ax.legend()
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ report
def render_report(*, name: str, config_path: Optional[str], fingerprint: str, validation: dict,
                  reference: Optional[dict], train: dict, comparison: str,
                  probes: Optional[str]) -> str:
    s = train["summary"]
    lines = [f"# GRPO run report: `{name}`", "",
             f"- config: `{config_path or '(in-memory)'}`; training fingerprint `{fingerprint}`",
             f"- {s['steps']} steps in {train['minutes']:.1f} min",
             "", "## Validation", "",
             "- gradient-direction correlation (tiny CPU model): "
             f"**{validation['grad_corr']:.4f}**",
             "- clip fraction across inner epochs 1..4 (tiny CPU model): "
             f"`{validation['clip_fracs']}`"]
    if reference:
        lines.append(f"- reference vs trained policy: mean |diff log p| = "
                     f"{reference['mean_abs_diff']:.3e}, KL = {reference['kl']:.5f}")
    lines += ["", "## Training", "",
              f"- degenerate groups {s['degenerate_rate']:.1%}; final-window KL {s['kl_last']:.4f}",
              f"- exact correctness (sampled, first vs last window): {s['exact_first']:.3f} -> "
              f"{s['exact_last']:.3f}; format {s['format_first']:.3f} -> {s['format_last']:.3f}",
              f"- clip fraction {s['clip_frac_mean']:.4f}; max |rho-1| at the first inner epoch "
              f"{s['ratio_dev_max']:.2e}",
              "- training windows contain different questions, so window-to-window changes mix "
              "learning with question difficulty; only the held-out comparison is paired.",
              "", "## Before vs after", "", "```", comparison, "```"]
    if probes:
        lines += ["", "## Question-level probe", "", "```", probes, "```"]
    return "\n".join(lines) + "\n"
