===============
The Greedy Path
===============

Another way to find a path is to choose the best pair to contract at every iteration so that the formula scales like N^3 - functionality provided by :func:`~opt_einsum.paths.greedy`.
The "best" contraction pair is currently determined by the smallest of the tuple ``(-removed_size, cost)`` where ``removed_size`` is the size of the contracted tensors minus the size of the tensor created and ``cost`` is the cost of the contraction.
Basically, we want to remove the largest dimensions at the least cost.
To prevent large outer products the results are sieved by the amount of memory available.
Overall, this turns out to work extremely well and is only slower than the optimal path in several cases, and even then only by a factor of 2-4 while only taking 1 millisecond for terms of length 10.
To me, while still not perfect, it represents a "good enough" algorithm for general production.
It is fast enough that at worst case the overhead penalty is approximately 20 microseconds and is much faster for every other einsum test case I can build or generate randomly.
