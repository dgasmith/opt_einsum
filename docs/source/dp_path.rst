============================
The Dynamic Programming Path
============================

The dynamic programming (DP) approach described in reference [1] provides an efficient
way to find an asymptotically optimal contraction path by running the following steps:

  1. Compute all traces, i.e. summations over indices occurring exactly in one
     input.
  2. Decompose the contraction graph of inputs into disconnected subgraphs. Two
     inputs are connected if they share at least one summation index.
  3. Find the contraction path for each of the disconnected subgraphs using a
     DP approach: The optimal contraction path for all sets of ``n`` (ranging
     from 1 to the number of inputs) connected tensors is found by combining
     sets of ``m`` and ``n-m`` tensors.

Note that computing all the traces in the very beginning can never lead to a
non-optimal contraction path.

Contractions of disconnected subgraphs can be optimized independently, which
still results in an optimal contraction path. However, the computational
complexity of finding the contraction path is drastically reduced: If the
subgraphs consist of ``n1``, ``n2``, ... inputs, the computational complexity
is reduced from ``O(exp(n1 + n2 + ...))`` to ``O(exp(n1) + exp(n2) + ...)``.

The DP approach will only perform pair contractions and will never compute
intermediate outer products as in reference [1] it is shown that this always
results in an asymptotically optimal contraction path.

A major optimization for DP is the cost capping strategy: The DP optimization
only memorizes contractions for a subset of inputs, if the total cost for this
contraction is smaller than the cost cap. The cost cap is initialized with
the minimal possible cost, i.e. the product of all output dimensions, and is
iteratively increased by multiplying it with the smallest dimension
until a contraction path including all inputs is found.

Note that the worst case scaling of DP is exponential in the number
of inputs. Nevertheless, if the contraction graph is not completely random,
but exhibits a certain kind of structure, it can be used for large
contraction graphs and is guaranteed to find an asymptotically optimal
contraction path. For this reason it is the most frequently used contraction
path optimizer in the field of tensor network states.


[1] Robert N. C. Pfeifer, Jutho Haegeman, and Frank Verstraete Phys. Rev. E 90, 033315 (2014). https://arxiv.org/abs/1304.6112
