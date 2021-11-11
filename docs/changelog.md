Changelog
=========

## 3.3.0 / 2020-07-19

Adds a `object` backend for optimized contractions on arbitrary Python objects.

**New Features**

 - [\#145](https://github.com/dgasmith/opt_einsum/pull/145) Adds a `object` based backend so that `contract(backend='object')` can be used on arbitrary objects such as SymPy symbols.

**Enhancements**

 - [\#140](https://github.com/dgasmith/opt_einsum/pull/140) Better error messages when the requested `contract` backend cannot be found.
 - [\#141](https://github.com/dgasmith/opt_einsum/pull/141) Adds a check with RandomOptimizers to ensure the objects are not accidentally reused for different contractions.
 - [\#149](https://github.com/dgasmith/opt_einsum/pull/149) Limits the `remaining` category for the `contract_path` output to only show up to 20 tensors to prevent issues with the quadratically scaling memory requirements and the number of print lines for large contractions.

## 3.2.0 / 2020-03-01

Small fixes for the `dp` path and support for a new mars backend.

**New Features**

 - [\#109](https://github.com/dgasmith/opt_einsum/pull/109) Adds mars backend support.

**Enhancements**

 - [\#110](https://github.com/dgasmith/opt_einsum/pull/110) New `auto-hq` and `'random-greedy-128'` paths.
 - [\#119](https://github.com/dgasmith/opt_einsum/pull/119) Fixes several edge cases in the `dp` path.

**Bug fixes**

 - [\#127](https://github.com/dgasmith/opt_einsum/pull/127) Fixes an issue where Python 3.6 features are required while Python 3.5 is `opt_einsum`'s stated minimum version.

## 3.1.0 / 2019-09-30

Adds a new dynamic programming algorithm to the suite of paths.

**New Features**

 - [\#102](https://github.com/dgasmith/opt_einsum/pull/102) Adds new `dp` path.

## 3.0.0 / 2019-08-10

This release moves `opt_einsum` to be backend agnostic while adding support
additional backends such as Jax and Autograd. Support for Python 2.7 has been dropped and Python 3.5 will become the new minimum version, a Python deprecation policy equivalent to NumPy's has been adopted.


**New Features**

- [\#78](https://github.com/dgasmith/opt_einsum/pull/78) A new random-optimizer has been implemented which uses Boltzmann weighting to explore alternative near-minimum paths using greedy-like schemes. This provides a fairly large path performance enhancements with a linear path time overhead.
- [\#78](https://github.com/dgasmith/opt_einsum/pull/78) A new PathOptimizer class has been implemented to provide a framework for building new optimizers. An example is that now custom cost functions can now be provided in the greedy formalism for building custom optimizers without a large amount of additional code.
- [\#81](https://github.com/dgasmith/opt_einsum/pull/81) The `backend="auto"` keyword has been implemented for `contract` allowing automatic detection of the correct backend to use based off provided tensors in the contraction.
- [\#88](https://github.com/dgasmith/opt_einsum/pull/88) Autograd and Jax support have been implemented.
- [\#96](https://github.com/dgasmith/opt_einsum/pull/96) Deprecates Python 2 functionality and devops improvements.

**Enhancements**

- [\#84](https://github.com/dgasmith/opt_einsum/pull/84) The `contract_path` function can now accept shape tuples rather than full tensors.
- [\#84](https://github.com/dgasmith/opt_einsum/pull/84) The `contract_path` automated path algorithm decision technology has been refactored to a standalone function.


## 2.3.0 / 2018-12-01

This release primarily focuses on expanding the suite of available path
technologies to provide better optimization characistics for 4-20 tensors while
decreasing the time to find paths for 50-200+ tensors. See `Path Overview <path_finding.html#performance-comparison>`_ for more information.

**New Features**

- [\#60](https://github.com/dgasmith/opt_einsum/pull/60) A new `greedy` implementation has been added which is up to two orders of magnitude faster for 200 tensors.
- [\#73](https://github.com/dgasmith/opt_einsum/pull/73) Adds a new `branch` path that uses `greedy` ideas to prune the `optimal` exploration space to provide a better path than `greedy` at sub `optimal` cost.
- [\#73](https://github.com/dgasmith/opt_einsum/pull/73) Adds a new `auto` keyword to the `opt_einsum.contract` `path` option. This keyword automatically chooses the best path technology that takes under 1ms to execute.

**Enhancements**

- [\#61](https://github.com/dgasmith/opt_einsum/pull/61) The `opt_einsum.contract` `path` keyword has been changed to `optimize` to more closely match NumPy. `path` will be deprecated in the future.
- [\#61](https://github.com/dgasmith/opt_einsum/pull/61) The `opt_einsum.contract_path` now returns a `opt_einsum.contract.PathInfo` object that can be queried for the scaling, flops, and intermediates of the path. The print representation of this object is identical to before.
- [\#61](https://github.com/dgasmith/opt_einsum/pull/61) The default `memory_limit` is now unlimited by default based on community feedback.
- [\#66](https://github.com/dgasmith/opt_einsum/pull/66) The Torch backend will now use `tensordot` when using a version of Torch which includes this functionality.
- [\#68](https://github.com/dgasmith/opt_einsum/pull/68) Indices can now be any hashable object when provided in the `"Interleaved Input" <input_format.html#interleaved-input>`_ syntax.
- [\#74](https://github.com/dgasmith/opt_einsum/pull/74) Allows the default `transpose` operation to be overridden to take advantage of more advanced tensor transpose libraries.
- [\#73](https://github.com/dgasmith/opt_einsum/pull/73) The `optimal` path is now significantly faster.
- [\#81](https://github.com/dgasmith/opt_einsum/pull/81) A documentation pass for v3.0.

**Bug fixes**

- [\#72](https://github.com/dgasmith/opt_einsum/pull/72) Fixes the `"Interleaved Input" <input_format.html#interleaved-input>`_ syntax and adds documentation.

## 2.2.0 / 2018-07-29

**New Features**

- [\#48](https://github.com/dgasmith/opt_einsum/pull/48) Intermediates can now be shared between contractions, see here for more details.
- [\#53](https://github.com/dgasmith/opt_einsum/pull/53) Intermediate caching is thread safe.

**Enhancements**

- [\#48](https://github.com/dgasmith/opt_einsum/pull/48) Expressions are now mapped to non-unicode index set so that unicode input is support for all backends.
- [\#54](https://github.com/dgasmith/opt_einsum/pull/54) General documentation update.

**Bug fixes**

- [\#41](https://github.com/dgasmith/opt_einsum/pull/41) PyTorch indices are mapped back to a small a-z subset valid for PyTorch's einsum implementation.

## 2.1.3 / 2018-8-23

**Bug fixes**

- Fixes unicode issue for large numbers of tensors in Python 2.7.
- Fixes unicode install bug in README.md.

## 2.1.2 / 2018-8-16

**Bug fixes**

- Ensures `versioneer.py` is in MANIFEST.in for a clean pip install.


## 2.1.1 / 2018-8-15

**Bug fixes**

- Corrected Markdown display on PyPi.

## 2.1.0 / 2018-8-15

`opt_einsum` continues to improve its support for additional backends beyond NumPy with PyTorch.

We have also published the opt_einsum package in the Journal of Open Source Software. If you use this package in your work, please consider citing us!

**New features**

- PyTorch backend support
- Tensorflow eager-mode execution backend support

**Enhancements**

- Intermediate tensordot-like expressions are now ordered to avoid transposes.
- CI now uses conda backend to better support GPU and tensor libraries.
- Now accepts arbitrary unicode indices rather than a subset.
- New auto path option which switches between optimal and greedy at four tensors.

**Bug fixes**

- Fixed issue where broadcast indices were incorrectly locked out of tensordot-like evaluations even after their dimension was broadcast.

## 2.0.1 / 2018-6-28

**New Features**

- Allows unlimited Unicode indices.
- Adds a Journal of Open-Source Software paper.
- Minor documentation improvements.


## 2.0.0 / 2018-5-17

`opt_einsum` is a powerful tensor contraction order optimizer for NumPy and related ecosystems.

**New Features**

- Expressions can be precompiled so that the expression optimization need not happen multiple times.
- The greedy order optimization algorithm has been tuned to be able to handle hundreds of tensors in several seconds.
- Input indices can now be unicode so that expressions can have many thousands of indices.
- GPU and distributed computing backends have been added such as Dask, TensorFlow, CUPy, Theano, and Sparse.

**Bug Fixes**

- An error affecting cases where opt_einsum mistook broadcasting operations for matrix multiply has been fixed.
- Most error messages are now more expressive.


## 1.0.0 / 2016-10-14

Einsum is a very powerful function for contracting tensors of arbitrary
dimension and index. However, it is only optimized to contract two terms at a
time resulting in non-optimal scaling for contractions with many terms.
Opt_einsum aims to fix this by optimizing the contraction order which can lead
to arbitrarily large speed ups at the cost of additional intermediate tensors.

Opt_einsum is also implemented into the np.einsum function as of NumPy v1.12.

**New Features**

- Tensor contraction order optimizer.
- `opt_einsum.contract` as a drop-in replacement for `numpy.einsum`.
