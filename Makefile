.DEFAULT_GOAL := all

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

.PHONY: mypy
mypy:
	mypy opt_einsum

.PHONY: test
test:
	pytest -v --cov=opt_einsum/

.PHONY: docs
docs:
	mkdocs build
