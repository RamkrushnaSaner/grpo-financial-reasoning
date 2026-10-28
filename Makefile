.PHONY: help install install-gpu test lint validate check mutation smoke reproduce base gated ablations clean

help:               ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install:            ## CPU: package + dev tools (tests, validation, CPU smoke run)
	pip install -e ".[dev]"

install-gpu:        ## GPU: everything needed to train
	pip install -e ".[train,dev]"

test:               ## CPU test suite, no model download
	python -m pytest

lint:
	ruff check src tests tools

validate:           ## reward properties + GRPO math checks (CPU, seconds)
	grpo validate

check:              ## full pipeline on the tiny CPU model (seconds; numbers meaningless)
	grpo run configs/cpu_smoke.yaml

mutation:           ## reintroduce every historical bug; the suite must fail for each (~10 min)
	python tools/mutation_test.py

smoke:              ## GPU, ~5 min: every stage on the real model
	grpo run configs/smoke.yaml

reproduce:          ## GPU: the reported run (Run 6), dataset v1
	grpo run configs/run6.yaml

base:               ## GPU: Run 6 method on the corrected dataset (v2)
	grpo run configs/base.yaml

gated:              ## GPU: gated reward + no dropout (unvalidated fix)
	grpo run configs/gated.yaml

ablations:          ## GPU: base + reward and lambda ablations, then a summary table
	grpo sweep configs/base.yaml configs/ablations/*.yaml --no-probes

clean:
	rm -rf results build dist .pytest_cache .ruff_cache src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
