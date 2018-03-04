================
The Optimal Path
================

The most optimal path can be found by searching through every possible way to contract the tensors together, this includes all combinations with the new intermediate tensors as well.
While this algorithm scales like N! and can often become more costly to compute than the unoptimized contraction itself, it provides an excellent benchmark.
The function that computes this path in opt_einsum is called :func:`~opt_einsum.paths.optimal` and works by iteratively finding every possible combination of pairs to contract in the current list of tensors.
This is iterated until all tensors are contracted together. The resulting paths are then sorted by total flop cost and the lowest one is chosen.
This algorithm runs in about 1 second for 7 terms, 15 seconds for 8 terms, and 480 seconds for 9 terms limiting its overall usefulness for a large number of terms.
By considering limited memory this can be sieved and can reduce the cost of computing the optimal function by an order of magnitude or more.

Lets look at an example:

.. code:: python

    Contraction:  abc,dc,ac->bd

Build a list with tuples that have the following form:

.. code:: python

    iteration 0:
     "(cost, path,  list of input sets remaining)"
    [ (0,    [],    [set(['a', 'c', 'b']), set(['d', 'c']), set(['a', 'c'])] ]

Since this is iteration zero, we have the initial list of input sets.
We can consider three possible combinations where we contract list positions (0, 1), (0, 2), or (1, 2) together:

.. code:: python

    iteration 1:
    [ (9504, [(0, 1)], [set(['a', 'c']), set(['a', 'c', 'b', 'd'])  ]),
      (1584, [(0, 2)], [set(['c', 'd']), set(['c', 'b'])            ]),
      (864,  [(1, 2)], [set(['a', 'c', 'b']), set(['a', 'c', 'd'])  ])]

We have now run through the three possible combinations, computed the cost of the contraction up to this point, and appended the resulting indices from the contraction to the list.
As all contractions only have two remaining input sets the only possible contraction is (0, 1):

.. code:: python

    iteration 2:
    [ (28512, [(0, 1), (0, 1)], [set(['b', 'd'])  ]),
      (3168,  [(0, 2), (0, 1)], [set(['b', 'd'])  ]),
      (19872, [(1, 2), (0, 1)], [set(['b', 'd'])  ])]

The final contraction cost is computed and we choose the second path from the list as the overall cost is the lowest.
