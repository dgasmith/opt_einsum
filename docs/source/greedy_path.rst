===============
The Greedy Path
===============

The ``greedy`` path iterates through the possible pair contractions and chooses the "best" contraction at every step until all contractions are considered.
The "best" contraction pair is determined by the smallest of the tuple ``(-removed_size, cost)`` where ``removed_size`` is the size of the contracted tensors minus the size of the tensor created and ``cost`` is the cost of the contraction.
Effectively, the algorithm chooses the best inner or dot product, Hadamard product, and then outer product at each iteration with a sieve to prevent large outer products.
This algorithm has proven to be quite successful for general production and only misses a few complex cases that make it slightly worse than the ``optimal`` algorithm.
Fortunately, these often only lead to increases in prefactor than missing the optimal scaling. 

The ``greedy`` scale like N^2 rather than factorially making ``greedy`` much more suitable for large numbers of contractions and has a lower prefactor that helps decrease latency.
As :mod:`opt_einsum` can handle more than a thousand unique indices the low scaling is especially important for very large contraction networks.
The ``greedy`` functionality is provided by :func:`~opt_einsum.paths.greedy`.
