# Release Checklist

### PyPI Source and Wheel:
```
python setup.py sdist
python setup.py bdist_wheel --universal
twine upload dist/*
```

### Update conda-forge:
 - Version
 - Zip Hash
