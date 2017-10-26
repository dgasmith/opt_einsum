==================
Examining the Path
==================

Now, lets consider the following expression found in a perturbation theory (one of ~5,000 such expressions):
.. code:: python

    bdik,acaj,ikab,ajac,ikbd

At first, it would appear that this scales like N^7 as there are 7 unique indices; however, we can define a intermediate to reduce this scaling.

.. code:: python

    a = bdik,ikab,ikbd` (N^5 scaling)

    result = acaj,ajac,a` (N^4 scaling)

This is a single possible path to the final answer (and notably, not the most optimal) out of many possible paths. Now, let opt_einsum compute the optimal path:

.. code:: python

    import opt_einsum as oe
    
    # Take a complex string
    einsum_string = 'bdik,acaj,ikab,ajac,ikbd->'
    
    # Build random views to represent this contraction
    unique_inds = set(einsum_string.replace(',', ''))
    index_size = [10, 17, 9, 10, 13, 16, 15, 14]
    sizes_dict = {c : s for c, s in zip(set(einsum_string), index_size)}
    views = oe.helpers.build_views(einsum_string, sizes_dict)
    
    path_info = oe.contract_path(einsum_string, *views)
    >>> print path_info[0]
    [(1, 3), (0, 2), (0, 2), (0, 1)]
    
    ```
    ```
    >>> print path_info[1]
      Complete contraction:  bdik,acaj,ikab,ajac,ikbd->
             Naive scaling:  7
         Optimized scaling:  4
          Naive FLOP count:  3.819e+08
      Optimized FLOP count:  8.000e+04
       Theoretical speedup:  4773.600
      Largest intermediate:  1.872e+03 elements
    --------------------------------------------------------------------------------
    scaling   BLAS                  current                                remaining
    --------------------------------------------------------------------------------
       3     False             ajac,acaj->a                       bdik,ikab,ikbd,a->
       4     False           ikbd,bdik->bik                             ikab,a,bik->
       4     False              bik,ikab->a                                    a,a->
       1       DOT                    a,a->                                       ->

.. code:: python

    einsum_result = np.einsum("bdik,acaj,ikab,ajac,ikbd->", *views)
    contract_result = contract("bdik,acaj,ikab,ajac,ikbd->", *views)
    >>> np.allclose(einsum_result, contract_result)
    True

By contracting terms in the correct order we can see that this expression can be computed with N^4 scaling. Even with the overhead of finding the best order or 'path' and small dimensions, opt_einsum is roughly 900 times faster than pure einsum for this expression.


=====================
More details on paths
=====================

Finding the optimal order of contraction is not an easy problem and formally scales factorially with respect to the number of terms in the expression. First, lets discuss what a path looks like in opt_einsum:

.. code:: python

    einsum_path = [(0, 1, 2, 3, 4)]
    opt_path = [(1, 3), (0, 2), (0, 2), (0, 1)]

In opt_einsum each element of the list represents a single contraction.
For example the einsum_path would effectively compute the result in a way identical to that of einsum itself, while the
opt_path would perform four contractions that form an identical result.
This opt_path represents the path taken in our above example.
The first contraction (1,3) contracts the first and third terms together to produce a new term which is then appended to the list of terms, this is continued until all terms are contracted.
An example should illuminate this:

.. code:: python

    ---------------------------------------------------------------------------------
    scaling   GEMM                   current                                remaining
    ---------------------------------------------------------------------------------
    terms = ['bdik', 'acaj', 'ikab', 'ajac', 'ikbd'] contraction = (1, 3)
      3     False              ajac,acaj->a                       bdik,ikab,ikbd,a->
    terms = ['bdik', 'ikab', 'ikbd', 'a'] contraction = (0, 2)
      4     False            ikbd,bdik->bik                             ikab,a,bik->
    terms = ['ikab', 'a', 'bik'] contraction = (0, 2)
      4     False              bik,ikab->a                                    a,a->
    terms = ['a', 'a'] contraction = (0, 1)
      1       DOT                    a,a->                                       ->
