# Release Checklist

### PyPI Source and Wheel:
```
conda update setuptools wheel

python setup.py sdist
twine upload --repository-url https://test.pypi.org/legacy/ dist/
```

### Update conda-forge:
 - Version
 - Zip Hash
