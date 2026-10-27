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

The Docs workflow keeps one rolling `dev` version from main and publishes each
final release from its tag under its major/minor version, such as `3.5`. Patch
releases update that entry. Final releases update `latest`, the site's default.
Prerelease tags do not publish separate documentation. Until the first versioned
final release, the root keeps serving the existing unversioned site. Previous
versions stay on `gh-pages`; old releases are not rebuilt automatically.

After publishing, check the version selector, API reference, examples, and math.
The 3.5 documentation work is tracked in
[#267](https://github.com/dgasmith/opt_einsum/issues/267).
