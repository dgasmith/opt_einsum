import math
import time
import heapq
import random
import functools
from collections import deque

from . import helpers
from .paths import PathOptimizer, _ssa_optimize, _BETTER_FNS, ssa_to_linear, _calc_k12_flops


__all__ = ["RandomGreedy", "random_greedy"]


class RandomOptimizer(PathOptimizer):
    """Base class for running any random path finder that benefits
    from repeated calling, possibly in a parallel fashion. Custom random
    optimizers should subclass this, and the ``setup`` method should be
    implemented with the following signature::

        def setup(self, inputs, output, size_dict):
            # custom preparation here ...
            return trial_fn, trial_args

    Where ``trial_fn`` itself should have the signature::

        def trial_fn(r, *trial_args):
            # custom computation of path here
            return ssa_path, cost, size

    Where ``r`` is the run number and could for example be used to seed a
    random number generator. See ``RandomGreedy`` for an example.


    Parameters
    ----------
    max_repeats : int, optional
        The maximum number of repeat trials to have.
    max_time : float, optional
        The maximum amount of time to run the algorithm for.
    minimize : {'flops', 'size'}, optional
        Whether to favour paths that minimize the total estimated flop-count or
        the size of the largest intermediate created.
    executor : executor-pool like, optional
        An executor-pool to optionally parallelize repeat trials over. The pool
        should have an api matching those found in the python 3 standard library
        module ``concurrent.futures``. Namely, a ``submit`` method that returns
        ``Future`` objects, themselves with ``result`` and ``cancel`` methods.
    pre_dispatch : int, optional
        If using an ``executor``, how many jobs to pre-dispatch so as to avoid
        submitting all jobs at once. Should also be more than twice the number
        of workers to avoid under-subscription. Default: 128.

    Attributes
    ----------
    path : list[tuple[int]]
        The best path found so far.
    costs : list[int]
        The list of each trial's costs found so far.
    sizes : list[int]
        The list of each trial's largest intermediate size so far.

    See Also
    --------
    RandomGreedy
    """

    def __init__(self, max_repeats=32, max_time=None, minimize='flops', executor=None, pre_dispatch=128):

        if minimize not in ('flops', 'size'):
            raise ValueError("`minimize` should be one of {'flops', 'size'}.")

        self.max_repeats = max_repeats
        self.max_time = max_time
        self.minimize = minimize
        self.better = _BETTER_FNS[minimize]
        self.executor = executor
        self.pre_dispatch = pre_dispatch

        self.costs = []
        self.sizes = []
        self.best = {'flops': float('inf'), 'size': float('inf')}

    @property
    def path(self):
        """The best path found so far.
        """
        return ssa_to_linear(self.best['ssa_path'])

    def _gen_results_executor(self, repeats, trial_fn, args):
        """Lazily generate results from an executor without submitting all jobs at once.
        """
        self._futures = deque()

        for r in repeats:
            if len(self._futures) < self.pre_dispatch:
                self._futures.append(self.executor.submit(trial_fn, r, *args))
                continue
            yield self._futures.popleft().result()

        while self._futures:
            yield self._futures.popleft().result()

    def _cancel_futures(self):
        if self.executor is not None:
            for f in self._futures:
                f.cancel()

    def __call__(self, inputs, output, size_dict, memory_limit):
        # start a timer?
        if self.max_time is not None:
            t0 = time.time()

        trial_fn, trial_args = self.setup(inputs, output, size_dict)
        repeat0 = len(self.costs)
        repeats = range(repeat0, repeat0 + self.max_repeats)

        # create the trials lazily
        if self.executor is not None:
            trials = self._gen_results_executor(repeats, trial_fn, trial_args)
        else:
            trials = (trial_fn(r, *trial_args) for r in repeats)

        # assess the trials
        for ssa_path, cost, size in trials:

            # keep track of all costs and sizes
            self.costs.append(cost)
            self.sizes.append(size)

            # check if we have found a new best
            found_new_best = self.better(cost, size, self.best['flops'], self.best['size'])

            if found_new_best:
                self.best['flops'] = cost
                self.best['size'] = size
                self.best['ssa_path'] = ssa_path

            # check if we have run out of time
            if (self.max_time is not None) and (time.time() > t0 + self.max_time):
                break

        self._cancel_futures()
        return self.path


def thermal_chooser(queue, remaining, nbranch=8, temperature=1, rel_temperature=True):
    """A contraction 'chooser' that weights possible contractions using a
    Boltzmann distribution. Explicitly, given costs ``c_i`` (with ``c_0`` the
    smallest), the relative weights, ``w_i``, are computed as:

        w_i = exp( -(c_i - c_0) / temperature)

    Additionally, if ``rel_temperature`` is set, scale ``temperature`` by
    ``abs(c_0)`` to account for likely fluctuating cost magnitudes during the
    course of a contraction.

    Parameters
    ----------
    queue : list
        The heapified list of candidate contractions.
    remaining : dict[str, int]
        Mapping of remaining inputs' indices to the ssa id.
    temperature : float, optional
        When choosing a possible contraction, its relative probability will be
        proportional to ``exp(-cost / temperature)``. Thus the larger
        ``temperature`` is, the further random paths will stray from the normal
        'greedy' path. Conversely, if set to zero, only paths with exactly the
        same cost as the best at each step will be explored.
    rel_temperature : bool, optional
        Whether to normalize the ``temperature`` at each step to the scale of
        the best cost. This is generally beneficial as the magnitude of costs
        can vary significantly throughout a contraction.
    nbranch : int, optional
        How many potential paths to calculate probability for and choose from
        at each step.

    Returns
    -------
    cost, k1, k2, k12
    """
    n = 0
    choices = []
    while queue and n < nbranch:
        cost, k1, k2, k12 = heapq.heappop(queue)
        if k1 not in remaining or k2 not in remaining:
            continue  # candidate is obsolete
        choices.append((cost, k1, k2, k12))
        n += 1

    if n == 0:
        return None
    if n == 1:
        return choices[0]

    costs = [choice[0][0] for choice in choices]
    cmin = costs[0]

    # adjust by the overall scale to account for fluctuating absolute costs
    if rel_temperature:
        temperature *= max(1, abs(cmin))

    # compute relative probability for each potential contraction
    if temperature == 0.0:
        energies = [1 if cost == cmin else 0 for cost in costs]
    else:
        # shift by cmin for numerical reasons
        energies = [math.exp(-(c - cmin) / temperature) for c in costs]

    # randomly choose a contraction based on energies
    chosen, = random.choices(range(n), weights=energies, k=1)
    cost, k1, k2, k12 = choices.pop(chosen)

    # put the other choise back in the heap
    for other in choices:
        heapq.heappush(queue, other)

    return cost, k1, k2, k12


def _ssa_path_compute_cost(ssa_path, inputs, output, size_dict):
    """Compute the flops and max size of an ssa path.
    """
    inputs = list(map(frozenset, inputs))
    output = frozenset(output)
    remaining = set(range(len(inputs)))
    total_cost = 0
    max_size = 0

    for i, j in ssa_path:
        k12, flops12 = _calc_k12_flops(inputs, output, remaining, i, j, size_dict)
        remaining.discard(i)
        remaining.discard(j)
        remaining.add(len(inputs))
        inputs.append(k12)
        total_cost += flops12
        max_size = max(max_size, helpers.compute_size_by_dict(k12, size_dict))

    return total_cost, max_size


def _trial_greedy_ssa_path_and_cost(r, inputs, output, size_dict, choose_fn, cost_fn):
    """A single, repeatable, greedy trial run. Returns ``ssa_path`` and cost.
    """
    if r == 0:
        # always start with the standard greedy approach
        choose_fn = None
    else:
        random.seed(r)

    ssa_path = _ssa_optimize(inputs, output, size_dict, choose_fn, cost_fn)
    cost, size = _ssa_path_compute_cost(ssa_path, inputs, output, size_dict)

    return ssa_path, cost, size


class RandomGreedy(RandomOptimizer):
    """

    Parameters
    ----------
    cost_fn : callable, optional
        A function that returns a heuristic 'cost' of a potential contraction
        with which to sort candidates. Should have signature
        ``cost_fn(size12, size1, size2, k12, k1, k2)``.
    temperature : float, optional
        When choosing a possible contraction, its relative probability will be
        proportional to ``exp(-cost / temperature)``. Thus the larger
        ``temperature`` is, the further random paths will stray from the normal
        'greedy' path. Conversely, if set to zero, only paths with exactly the
        same cost as the best at each step will be explored.
    rel_temperature : bool, optional
        Whether to normalize the ``temperature`` at each step to the scale of
        the best cost. This is generally beneficial as the magnitude of costs
        can vary significantly throughout a contraction. If False, the
        algorithm will end up branching when the absolute cost is low, but
        stick to the 'greedy' path when the cost is high - this can also be
        beneficial.
    nbranch : int, optional
        How many potential paths to calculate probability for and choose from
        at each step.
    kwargs
        Supplied to RandomOptimizer.

    See Also
    --------
    RandomOptimizer
    """

    def __init__(self, cost_fn='memory-removed-jitter', temperature=1.0,
                 rel_temperature=True, nbranch=8, **kwargs):
        self.cost_fn = cost_fn
        self.temperature = temperature
        self.rel_temperature = rel_temperature
        self.nbranch = nbranch
        super().__init__(**kwargs)

    @property
    def choose_fn(self):
        """The function that chooses which contraction to take - make this a
        property so that ``temperature`` and ``nbranch`` etc. can be updated
        between runs.
        """
        if self.nbranch == 1:
            return None

        return functools.partial(thermal_chooser, temperature=self.temperature,
                                 nbranch=self.nbranch, rel_temperature=self.rel_temperature)

    def setup(self, inputs, output, size_dict):
        fn = _trial_greedy_ssa_path_and_cost
        args = (inputs, output, size_dict, self.choose_fn, self.cost_fn)
        return fn, args


def random_greedy(inputs, output, idx_dict, memory_limit=None, **optimizer_kwargs):
    """
    """
    optimizer = RandomGreedy(**optimizer_kwargs)
    return optimizer(inputs, output, idx_dict, memory_limit)
