================
The Optimal Path
================

The most optimal path can be found by searching through every possible way to contract the tensors together, this includes all combinations with the new intermediate tensors as well.
While this algorithm scales like N!, and can often become more costly to compute than the unoptimized contraction itself, it provides an excellent benchmark.
The function that computes this path in opt_einsum is called :func:`~opt_einsum.paths.optimal` and works by performing a recursive, depth-first search. By keeping track of the
best path found so far, in terms of total estimated FLOP count, the search can
then quickly prune many paths as soon as as they exceed this best.
This optimal strategy is used by default with the ``optimize='auto'`` mode of
``opt_einsum`` for 4 tensors or less, though it can handle expressions of up to
9-10 tensors in a matter of seconds.


Let us look at an example:

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

The final contraction cost is computed, and we choose the second path from the list as the overall cost is the lowest.
