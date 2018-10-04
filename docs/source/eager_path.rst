==============
The Eager Path
==============

The ``eager`` path is constructed in three stages:
1. Eagerly compute Hadamard products (in arbitrary order -- this is commutative).
2. Greedily contract pairs of remaining tensors, at each step choosing the pair that minimizes ``result_size - sum(input_sizes) / 2``.
3. Greedily compute pairwise outer products, at each step choosing the pair that minimizes ``sum(input_sizes)``.

The ``eager`` has space and time complexity ``O(n * k)`` where ``n`` is the number of tensors and ``k`` is the maximum number of tensors that share any dimension (excluding dimensions that occurr in the output or in every tensor).
This algorithm this scales well to very large sparse contractions of small tensors.
The ``eager`` functionality is provided by :func:`~opt_einsum.paths.eager`.
