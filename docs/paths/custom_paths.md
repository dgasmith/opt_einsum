# Custom Path Optimizers

If you want to implement or just experiment with custom contaction paths then
you can easily by subclassing the [`opt_einsum.paths.PathOptimizer`](../api_reference.md#opt_einsumpathspathoptimizer)
object. For example, imagine we want to test the path that just blindly
contracts the first pair of tensors again and again. We would implement this
as:

```python
import opt_einsum as oe

class MyOptimizer(oe.paths.PathOptimizer):

    def __call__(self, inputs, output, size_dict, memory_limit=None):
        return [(0, 1)] * (len(inputs) - 1)
```

Once defined we can use this as:

```python
import numpy as np

# set-up a random contraction
eq, shapes = oe.helpers.rand_equation(10, 3, seed=42)
arrays = list(map(np.ones, shapes))

# set-up our optimizer and use it
optimizer = MyOptimizer()
path, path_info = oe.contract_path(eq, *arrays, optimize=optimizer)

print(path)
#> [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]

print(path_info.speedup)
#> 133.21363671496357
```

Note that though we still get a considerable speedup over `einsum` this is
of course not a good strategy to take in general.


## Custom Random Optimizers

If your custom path optimizer is inherently random, then you can reuse all the
machinery of the random-greedy approach. Namely:

- A **max-repeats** or **max-time** approach
- Minimization with respect to total flops or largest intermediate size
- Parallelization using a pool-executor

This is done by subclassing the
[`opt_einsum.paths.RandomOptimizer`](../api_reference.md#opt_einsumpathsrandomoptimizer)
 object and implementing a
`setup` method. Here's an example where we just randomly select any path
(again, although we get a considerable speedup over `einsum` this is
not a good strategy to take in general):

```python
from opt_einsum.path_random import ssa_path_compute_cost

class MyRandomOptimizer(oe.path_random.RandomOptimizer):

    @staticmethod
    def random_path(r, n, inputs, output, size_dict):
        """Picks a completely random contraction order.
        """
        np.random.seed(r)
        ssa_path = []
        remaining = set(range(n))
        while len(remaining) > 1:
            i, j = np.random.choice(list(remaining), size=2, replace=False)
            remaining.add(n + len(ssa_path))
            remaining.remove(i)
            remaining.remove(j)
            ssa_path.append((i, j))
        cost, size = ssa_path_compute_cost(ssa_path, inputs, output, size_dict)
        return ssa_path, cost, size

    def setup(self, inputs, output, size_dict):
        """Prepares the function and arguments to repeatedly call.
        """
        n = len(inputs)
        trial_fn = self.random_path
        trial_args = (n, inputs, output, size_dict)
        return trial_fn, trial_args
```

Which we can now instantiate using various other options:

```python
optimizer = MyRandomOptimizer(max_repeats=1000, max_time=10,
                              parallel=True, minimize='size')
path, path_info = oe.contract_path(eq, *arrays, optimize=optimizer)

print(path)
#> [(3, 4), (1, 3), (0, 3), (3, 5), (3, 4), (3, 4), (1, 0), (0, 1), (0, 1)]

print(path_info.speedup)
#> 712829.9451056132
```

There are a few things to note here:

1. The core function (`MyRandomOptimizer.random_path` here), should take a
   trial number `r` as it first argument
2. It should return a *ssa_path* (see `opt_einsum.paths.ssa_to_linear` and
   `opt_einsum.paths.linear_to_ssa`) as well as a flops-cost and max-size.
3. The `setup` method prepares this function, as well as any input to it,
   so that the trials will look roughly like
   `[trial_fn(r, *trial_args) for r in range(max_repeats)]`. If you need to
   parse the standard arguments (into a network for example), it thus only
   needs to be done once per optimization

More details about
[`opt_einsum.paths.RandomOptimizer`](../api_reference.md#opt_einsumpath_randomrandomoptimizer)
options can
be found in [`RandomGreedyPathPage`](./random_greedy_path.md) section.
