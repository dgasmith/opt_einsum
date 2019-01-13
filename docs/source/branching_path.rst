==================
The Branching Path
==================

While the ``optimal`` path is guaranteed to find the smallest estimate FLOP
cost, it spends a lot of time exploring paths which are not likely to result in
an optimal path. For instance, outer products are usually not advantageous
unless absolutely necessary. Additionally, by trying a 'good' path first, it
should be possible to quickly establish a threshold FLOP cost which can then be
used to prune many bad paths.

The **branching** strategy (provided by :func:`~opt_einsum.paths.branch`) does
this by taking the recursive, depth-first approach of
:func:`~opt_einsum.paths.optimal`, whilst also sorting potential contractions
based on a heuristic cost, as in :func:`~opt_einsum.paths.greedy`.

There are two main flavours:

    - ``optimize='branch-all'``: explore **all** inner products, starting with
      those that look best according to the cost heuristic.
    - ``optimize='branch-2'``: similar, but at each step only explore the
      estimated best **two** possible contractions, leading to a maximum of
      2^N paths assessed.

In both cases, :func:`~opt_einsum.paths.branch` takes an active approach to
pruning paths well before they hit the best *total* FLOP count, by comparing
them to the FLOP count (times some factor) achieved by the best path at the
same point in the contraction.

There is also ``'branch-1'``, which, since it only explores a single path at
each step does not really 'branch' - this is essentially the approach of
``'greedy'``.
In comparison, ``'branch-1'`` will be slower for large expressions, but for
small to medium expressions it might find slightly higher quality contractions
due to considering individual flop costs at each step.

The default ``optimize='auto'`` mode of ``opt_einsum`` will use
``'branch-all'`` for 5 or 6 tensors, though it should be able to handle
12-13 tensors in a matter or seconds. Likewise, ``'branch-2'`` will be used for
7 or 8 tensors, though it should be able to handle 20-22 tensors in a matter of
seconds. Finally, ``'branch-1'`` will be used by ``'auto'`` for expressions of
up to 14 tensors.


Customizing the Branching Path
------------------------------

The 'branch and bound' path can be customized by creating a custom
:class:`~opt_einsum.paths.BranchBound` instance. For example:

.. code:: python

    optimizer = oe.BranchBound(nbranch=3, minimize='size', cutoff_flops_factor=None)
    path, path_info = oe.contract_path(eq, *arrays, optimize=optimizer)

You could then tweak the settings (e.g. ``optimizer.nbranch = 4``) and the best
bound found so far will persist and be used to prune paths on the next call:

.. code:: python

    optimizer.nbranch = 4
    path, path_info = oe.contract_path(eq, *arrays, optimize=optimizer)
