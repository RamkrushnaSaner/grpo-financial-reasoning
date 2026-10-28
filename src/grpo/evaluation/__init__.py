"""Evaluation: held-out accuracy with per-question records, the base-vs-trained
probe, paired before/after statistics, plots and the run report."""
from .heldout import evaluate, summarize_records
from .probes import ask_both, render_ab, run_probes, summarize_probes
from .report import (
    compare,
    format_comparison,
    plot_before_after,
    plot_training,
    render_report,
)

__all__ = ["ask_both", "compare", "evaluate", "format_comparison", "plot_before_after",
           "plot_training", "render_ab", "render_report", "run_probes", "summarize_probes",
           "summarize_records"]
