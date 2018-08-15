Changelog
=========

2.2.0 / 2018-MM-DD
------------------

New Features
++++++++++++

- Improved documentation.


2.1.1 / 2018-8-16
-----------------

Bug fixes
+++++++++

- Corrected Markdown display on PyPi.

2.1.0 / 2018-8-15
-----------------

``opt_einsum`` continues to improve its support for additional backends beyond NumPy with PyTorch.

We have also published the opt_einsum package in the Journal of Open Source Software. If you use this package in your work, please consider citing us!

New features
++++++++++++

- PyTorch backend support
- Tensorflow eager-mode execution backend support

Enhancements
++++++++++++

- Intermediate tensordot-like expressions are now ordered to avoid transposes.
- CI now uses conda backend to better support GPU and tensor libraries.
- Now accepts arbitrary unicode indices rather than a subset.
- New auto path option which switches between optimal and greedy at four tensors.

Bug fixes
+++++++++

- Fixed issue where broadcast indices were incorrectly locked out of tensordot-like evaluations even after their dimension was broadcast.

2.0.1 / 2018-6-28
-----------------

New Features
++++++++++++

- Allows unlimited Unicode indices.
- Adds a Journal of Open-Source Software paper.
- Minor documentation improvements.


2.0.0 / 2018-5-17
-----------------

``opt_einsum`` is a powerful tensor contraction order optimizer for NumPy and related ecosystems.

New Features
++++++++++++

- Expressions can be precompiled so that the expression optimization need not happen multiple times.
- The greedy order optimization algorithm has been tuned to be able to handle hundreds of tensors in several seconds.
- Input indices can now be unicode so that expressions can have many thousands of indices.
- GPU and distributed computing backends have been added such as Dask, TensorFlow, CUPy, Theano, and Sparse.

Bug Fixes
+++++++++

- A error effecting cases where opt_einsum mistook broadcasting operations for matrix multiply has been fixed.
- Most error messages are now more expressive.


1.0.0 / 2016-10-14
------------------

Einsum is a very powerful function for contracting tensors of arbitrary
dimension and index. However, it is only optimized to contract two terms at a
time resulting in non-optimal scaling for contractions with many terms.
Opt_einsum aims to fix this by optimizing the contraction order which can lead
to arbitrarily large speed ups at the cost of additional intermediate tensors.

Opt_einsum is also implemented into the np.einsum function as of NumPy v1.12.

New Features
++++++++++++

- Tensor contraction order optimizer.
- :func:`opt_einsum.contract` as a drop-in replacement for :func:`numpy.einsum`.
