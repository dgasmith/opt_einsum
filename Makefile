.DEFAULT_GOAL := all

.PHONY: install
install:
	pip install -e .

.PHONY: format
format:
	ruff check opt_einsum --fix
	ruff format opt_einsum

.PHONY: format-check
format-check:
	ruff check opt_einsum
	ruff format --check opt_einsum

.PHONY: check-dist
check-dist:
	python setup.py check -ms
	python setup.py sdist
	twine check dist/*

.PHONY: mypy
mypy:
	$(mypy)

.PHONY: test
test:
	pytest -v --cov=opt_einsum/

.PHONY: docs
docs:
	mkdocs build
