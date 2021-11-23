.DEFAULT_GOAL := all
isort = isort opt_einsum scripts/
black = black opt_einsum scripts/
autoflake = autoflake -ir --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables opt_einsum scripts/
mypy = mypy --ignore-missing-imports codex opt_einsum scripts/

.PHONY: install
install:
	pip install -e .

.PHONY: format
format:
	$(autoflake)
	$(isort)
	$(black)

.PHONY: format-check
format-check:
	$(isort) --check-only
	$(black) --check

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
