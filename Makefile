.DEFAULT_GOAL := all

PYTHON ?= 3.12
PYTEST_ARGS ?= -v --cov=opt_einsum/ --cov-report=term --cov-report=xml
TEST_RUN = UV_PROJECT_ENVIRONMENT=.cache/test-envs/$(PYTHON)-$@ \
	uv run --python $(PYTHON) --exact --extra test

.PHONY: format
format:
	uv run --with ".[lint]" ruff check opt_einsum --fix
	uv run --with ".[lint]" ruff format opt_einsum

.PHONY: format-check
format-check:
	uv run --with ".[lint]" ruff check opt_einsum
	uv run --with ".[lint]" ruff format --check opt_einsum

.PHONY: mypy
mypy:
	uv run --with ".[test,lint]" mypy opt_einsum

.PHONY: test test-full test-core test-numpy test-torch test-backends
test: test-core

test-core:
	@$(TEST_RUN) pytest $(PYTEST_ARGS)

test-numpy:
	@$(TEST_RUN) --extra numpy pytest $(PYTEST_ARGS)

test-torch:
	@$(TEST_RUN) --extra torch pytest $(PYTEST_ARGS)

test-backends:
	@$(TEST_RUN) --extra numpy --extra torch --extra dask --extra jax --extra sparse pytest $(PYTEST_ARGS)

# Run sequentially because the suites write the same coverage files.
test-full:
	@$(MAKE) --no-print-directory --keep-going -j1 test-core test-numpy test-torch test-backends

.PHONY: docs
docs:
	uv run --with ".[docs]" mkdocs build --strict

.PHONY: docs-serve
docs-serve:
	uv run --with ".[docs]" mkdocs serve
