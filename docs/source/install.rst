Install opt_einsum
==================

You can install opt_einsum with ``conda``, with ``pip``, or by installing from source.

Conda
-----

You can update opt_einsum using `conda <https://www.anaconda.com/download/>`_::

    conda install opt_einsum -c conda-forge

This installs opt_einsum and the NumPy dependancy.

The opt_einsum package is maintained on the
`conda-forge channel <https://conda-forge.github.io/>`_.


Pip
---

To install opt_einsum with ``pip`` there are a few options, depending on which
dependencies you would like to keep up to date:

*   ``pip install opt_einsum``

Install from Source
-------------------

To install opt_einsum from source, clone the repository from `github
<https://github.com/dgasmith/opt_einsum>`_::

    git clone https://github.com/dgasmith/opt_einsum.git
    cd opt_einsum
    python setup.py install

or use ``pip`` locally if you want to install all dependencies as well::

    pip install -e .


Test
----

Test opt_einsum with ``py.test``::

    cd opt_einsum
    py.test
