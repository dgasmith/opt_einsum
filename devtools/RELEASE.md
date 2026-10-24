# Release Checklist

## Lint Static Scan

Check for flake8 issues and spelling.

```shell
pip install flake8-spellcheck
flake8 --whitelist ./devtools/allowlist.txt
```

## PyPI Source and Wheel

```shell
conda update setuptools wheel

python setup.py sdist bdist_wheel
twine upload --repository-url https://test.pypi.org/legacy/ dist/
```

## Update conda-forge

```plaintext
 - Version
 - Zip Hash
```

## Documentation

The Docs workflow publishes main as `dev` and each GitHub release from its tag.
Final releases update `stable`; prereleases leave it alone. The site root uses
`stable` once available, otherwise `dev`. Existing versions and unversioned
files stay on `gh-pages`; old releases are not rebuilt automatically.

After publishing, check the version selector, API reference, examples, and math.
The 3.5 documentation work is tracked in
[#267](https://github.com/dgasmith/opt_einsum/issues/267).

For a local publication check, run `make docs-publish` in a disposable clone.
Use `DOCS_VERSION=3.5.0 DOCS_ALIAS=stable` for a final release, or omit the alias
for a prerelease. Only `DOCS_FLAGS=--push` pushes the generated commits.
