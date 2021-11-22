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
