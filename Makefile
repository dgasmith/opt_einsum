.DEFAULT_GOAL := all

<<<<<<< HEAD
.PHONY: format
format:
	uv run ruff check opt_einsum --fix
	uv run ruff format opt_einsum

.PHONY: format-check
format-check:
	uv run ruff check opt_einsum
	uv run ruff format --check opt_einsum
=======
.PHONY: install
install:
	pip install -e .

.PHONY: fmt
fmt:
	ruff check opt_einsum --fix
	ruff format opt_einsum

.PHONY: fmt-unsafe
fmt-unsafe:
	ruff check opt_einsum --fix --unsafe-fixes
	ruff format opt_einsum

.PHONY: fmt-check
fmt-check:
	ruff check opt_einsum
	ruff format --check opt_einsum
>>>>>>> 70031398125a9d5c6de87e038c7c6a50a11dac79

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
