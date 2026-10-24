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
	uv run --with ".[test,lint]" mypy opt_einsum

.PHONY: test
test:
	uv run --with ".[test]" pytest -v --cov=opt_einsum/ --cov-report=xml

.PHONY: docs
docs:
	uv run --with ".[docs]" mkdocs build --strict

.PHONY: docs-serve
docs-serve:
	uv run --with ".[docs]" mkdocs serve

DOCS_VERSION ?= dev
DOCS_ALIAS ?=
DOCS_FLAGS ?=

.PHONY: docs-publish
docs-publish:
	uv run --with ".[docs]" mike deploy --update-aliases "$(DOCS_VERSION)" $(DOCS_ALIAS)
	@default="$(DOCS_VERSION)"; \
	for version in stable dev; do \
		if uv run --with ".[docs]" mike list "$$version" >/dev/null 2>&1; then \
			default="$$version"; break; \
		fi; \
	done; \
	uv run --with ".[docs]" mike set-default $(DOCS_FLAGS) "$$default"
