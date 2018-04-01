import setuptools
import versioneer

if __name__ == "__main__":
    setuptools.setup(
        name='opt_einsum',
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        description='Optimizing numpys einsum function',
        author='Daniel Smith',
        author_email='dgasmith@icloud.com',
        url="https://github.com/dgasmith/opt_einsum",
        license='MIT',
        packages=setuptools.find_packages(),
        install_requires=[
            'numpy>=1.7',
        ],
        extras_require={
            'docs': [
                'sphinx==1.2.3',  # autodoc was broken in 1.3.1
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
            'tests': [
                'pytest',
                'pytest-cov',
                'pytest-pep8',
            ],
        },

        tests_require=[
            'pytest',
            'pytest-cov',
            'pytest-pep8',
        ],

        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
        ],
        zip_safe=True,
        long_description="""
Einsum is a very powerful function for contracting tensors of arbitrary
dimension and index. However, it is only optimized to contract two terms
at a time resulting in non-optimal scaling.

For example, consider the following index transformation:
``M_{pqrs} = C_{pi} C_{qj} I_{ijkl} C_{rk} C_{sl}``

Consider two different algorithms:

.. code:: python

    import numpy as np
    N = 10
    C = np.random.rand(N, N)
    I = np.random.rand(N, N, N, N)

    def naive(I, C):
        # N^8 scaling
        return np.einsum('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)

    def optimized(I, C):
        # N^5 scaling
        K = np.einsum('pi,ijkl->pjkl', C, I)
        K = np.einsum('qj,pjkl->pqkl', C, K)
        K = np.einsum('rk,pqkl->pqrl', C, K)
        K = np.einsum('sl,pqrl->pqrs', C, K)
        return K

The einsum function does not consider building intermediate arrays;
therefore, helping einsum out by building these intermediate arrays can result
in a considerable cost savings even for small N (N=10):

.. code:: python

    >> np.allclose(naive(I, C), optimized(I, C))
    True

    %timeit naive(I, C)
    1 loops, best of 3: 1.18 s per loop

    %timeit optimized(I, C)
    1000 loops, best of 3: 612 µs per loop

The index transformation is a well known contraction that leads to
straightforward intermediates. This contraction can be further
complicated by considering that the shape of the C matrices need not be
the same, in this case the ordering in which the indices are transformed
matters greatly. Logic can be built that optimizes the ordering;
however, this is a lot of time and effort for a single expression.

The opt_einsum package is a drop in replacement for the ``np.einsum`` function
and can handle all of the logic for you:

.. code:: python

    from opt_einsum import contract

    contract('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)

The above will automatically find the optimal contraction order, in this case
identical to that of the optimized function above, and compute the products for
you. In this case, it even uses `np.dot` under the hood to exploit any vendor
BLAS functionality that your NumPy build has!

We can then view more details about the optimized contraction order:

.. code:: python

    >>> from opt_einsum import contract_path

    >>> path_info = oe.contract_path('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)

    >>> print(path_info[0])
    [(0, 2), (0, 3), (0, 2), (0, 1)]

    >>> print(path_info[1])
      Complete contraction:  pi,qj,ijkl,rk,sl->pqrs
             Naive scaling:  8
         Optimized scaling:  5
          Naive FLOP count:  8.000e+08
      Optimized FLOP count:  8.000e+05
       Theoretical speedup:  1000.000
      Largest intermediate:  1.000e+04 elements
    --------------------------------------------------------------------------------
    scaling   BLAS                  current                                remaining
    --------------------------------------------------------------------------------
       5      GEMM            ijkl,pi->jklp                      qj,rk,sl,jklp->pqrs
       5      GEMM            jklp,qj->klpq                         rk,sl,klpq->pqrs
       5      GEMM            klpq,rk->lpqr                            sl,lpqr->pqrs
       5      GEMM            lpqr,sl->pqrs                               pqrs->pqrs
"""
    )
