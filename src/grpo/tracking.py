"""Logging: a human-readable log per run and a machine-readable metrics stream.

* ``get_logger`` writes to stdout and to the run's ``run.log``. One run = one
  log file: attaching a new file detaches the previous one (two runs in one
  Colab session must produce two logs).
* ``MetricsWriter`` appends one JSON object per training step to
  ``metrics.jsonl``, so a run that dies at step 40 still has 40 steps of data,
  and any tool (pandas, a notebook, a plotting script) can read it.
"""
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union


class _CurrentStdoutHandler(logging.StreamHandler):
    """Writes to whatever sys.stdout is at emit time, so notebook cells and pytest
    capture both see the output (a handler bound at creation keeps writing to a
    stale stream)."""

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        pass


def get_logger(name: str = "grpo", log_file: Optional[Union[str, Path]] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    if not any(isinstance(h, _CurrentStdoutHandler) for h in logger.handlers):
        sh = _CurrentStdoutHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file is not None:
        path = str(Path(log_file).resolve())
        for h in list(logger.handlers):
            if isinstance(h, logging.FileHandler) and h.baseFilename != path:
                logger.removeHandler(h)
                h.close()
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(path)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def _jsonable(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


class MetricsWriter:
    """Append-only JSONL: one record per training step."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")                       # a fresh run starts a fresh stream

    def write(self, record: Dict[str, Any]) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(_jsonable(record)) + "\n")


def read_metrics(path: Union[str, Path]) -> list:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
