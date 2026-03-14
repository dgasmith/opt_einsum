.DEFAULT_GOAL := all

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
	uv run --with ".[lint]" mypy opt_einsum

.PHONY: test
test:
	uv run --with ".[test]" pytest -v --cov=opt_einsum/

.PHONY: docs
docs:
	uv run --with ".[docs]" mkdocs build
