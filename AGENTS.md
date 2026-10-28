# Working on opt_einsum

## Understand the project

Start with the [overview](docs/index.md), then use [mkdocs.yml](mkdocs.yml) to
find the guides and API documentation relevant to your change. Read them
alongside the implementation and nearby tests. Investigate disagreements
between documented and actual behavior before deciding what to change.

Keep the core dependency-free and preserve backend-native types and behavior.
Protect numerical correctness, and avoid significant performance regressions
in core search and execution paths. Explain significant runtime or memory tradeoffs.

## Design principles

- Make the smallest coherent change that solves the underlying problem.
- Make the code’s intent and behavior easy to follow.
- Keep existing names, interfaces and patterns when they still fit.
- Reuse existing mechanisms; new abstractions must justify their cost.
- Protect correctness and compatibility; account for performance and memory.

## Keep documentation accurate

Treat the docs as part of the codebase. Update relevant guides, examples and
docstrings when behavior or public interfaces change. Explain important
constraints and reasons where future contributors will look for them.
Leave README.md to human maintainers; raise any conflicts with them.

Run new or changed examples. Update `mkdocs.yml` when published navigation
changes. Build the docs and inspect rendered pages when their presentation
changes.

## Check your work

Check relevant existing issues and PRs for overlapping work when available.
For behavior changes, extend nearby tests where possible; use small cases and
distinct dimensions that expose the bug. Exercise affected backend behavior
with that backend installed. The test extra does not install array backends;
check which relevant tests actually ran and report skips.

Use uv through the existing Make targets, choosing checks appropriate to the
change:

- `make format-check`: lint and formatting
- `make mypy`: type checking
- `make test`: tests
- `make docs`: strict documentation build
- `make docs-serve`: local preview

Inspect the final diff and run `git diff --check`. Report what you checked,
failures and any behavior left unverified.

## Before publishing

State whether a human reviewed the diff; reviewing it together counts.
If nobody reviewed it, include `Human review: not reviewed`.
If you cannot confirm review, include `Human review: not confirmed`.
