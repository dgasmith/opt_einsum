.DEFAULT_GOAL := all

.PHONY: format
format:
	uv run ruff check opt_einsum --fix
	uv run ruff format opt_einsum

.PHONY: format-check
format-check:
	uv run ruff check opt_einsum
	uv run ruff format --check opt_einsum

.PHONY: mypy
mypy:
	uv run mypy opt_einsum

.PHONY: test
test:
	uv pip install -e ".[test]"
	uv run pytest -v --cov=opt_einsum/

.PHONY: docs
docs:
	uv run mkdocs build
