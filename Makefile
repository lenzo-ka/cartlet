# Prefer a local venv/ if present, otherwise fall back to tools on PATH.
# Override explicitly with e.g. `make PY=python3`.
PY ?= $(shell [ -x venv/bin/python ] && echo venv/bin/python || echo python3)
PYTEST ?= $(shell [ -x venv/bin/pytest ] && echo venv/bin/pytest || echo python3 -m pytest)
RUFF ?= $(shell [ -x venv/bin/ruff ] && echo venv/bin/ruff || echo ruff)
MYPY ?= $(shell [ -x venv/bin/mypy ] && echo venv/bin/mypy || echo mypy)

.PHONY: test
test:
	$(PYTEST) -q

.PHONY: lint
lint:
	$(RUFF) check .
	$(MYPY) cartlet/

.PHONY: format
format:
	$(RUFF) format .
	$(RUFF) check --fix .

.PHONY: bench
bench:
	$(PY) -m benchmarks.runner_throughput

.PHONY: bench-json
bench-json:
	$(PY) -m benchmarks.runner_throughput --json bench-results.json

.PHONY: examples
examples:
	$(PY) -m examples.iris_decision_tree
	$(PY) -m examples.wine_random_forest
	$(PY) -m examples.breast_cancer_binary
	$(PY) -m examples.diabetes_regression
	$(PY) -m examples.iris_runner_deploy

.PHONY: clean
clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
