# Release Checklist

### PyPI Source and Wheel:
```
conda update setuptools wheel

python3 setup.py sdist
twine upload --repository-url https://test.pypi.org/legacy/ sdist/
```

### Update conda-forge:
 - Version
 - Zip Hash
