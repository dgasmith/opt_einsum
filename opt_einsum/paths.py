"""
Contains the path technology behind opt_einsum in addition to several path helpers
"""
from __future__ import absolute_import, division, print_function

import math
import heapq
import random
import itertools
import functools
from collections import defaultdict

import numpy as np

from . import helpers

__all__ = ["optimal", "BranchOptimizer", "branch", "greedy", "RandomOptimizer", "random_greedy"]


_UNLIMITED_MEM = {-1, None, float('inf')}


class PathOptimizer(object):
    """Base class for different path optimizers to inherit from.

    Subclassed optimizers should define a call method with signature::

        def __call__(self, inputs, output, size_dict, memory_limit=None):
            # ... compute path here ...
            return path

    where ``path`` is a list of int-tuples specifiying a contraction order.
    """
    pass


def ssa_to_linear(ssa_path):
    """
    Convert a path with static single assignment ids to a path with recycled
    linear ids. For example::

        >>> ssa_to_linear([(0, 3), (2, 4), (1, 5)])
        [(0, 3), (1, 2), (0, 1)]
    """
    ids = np.arange(1 + max(map(max, ssa_path)), dtype=np.int32)
    path = []
    for ssa_ids in ssa_path:
        path.append(tuple(int(ids[ssa_id]) for ssa_id in ssa_ids))
        for ssa_id in ssa_ids:
            ids[ssa_id:] -= 1
    return path


def linear_to_ssa(path):
    """
    Convert a path with recycled linear ids to a path with static single
    assignment ids. For example::

        >>> linear_to_ssa([(0, 3), (1, 2), (0, 1)])
        [(0, 3), (2, 4), (1, 5)]
    """
    num_inputs = sum(map(len, path)) - len(path) + 1
    linear_to_ssa = list(range(num_inputs))
    new_ids = itertools.count(num_inputs)
    ssa_path = []
    for ids in path:
        ssa_path.append(tuple(linear_to_ssa[id_] for id_ in ids))
        for id_ in sorted(ids, reverse=True):
            del linear_to_ssa[id_]
        linear_to_ssa.append(next(new_ids))
    return ssa_path


def _calc_k12_flops(inputs, output, remaining, i, j, size_dict):
    """
    Calculate the resulting indices and flops for a potential pairwise
    contraction - used in the recursive (optimal/branch) algorithms.

    Parameters
    ----------
    inputs : tuple[frozenset[str]]
        The indices of each tensor in this contraction, note this includes
        tensors unavaiable to contract as static single assignment is used ->
        contracted tensors are not removed from the list.
    output : frozenset[str]
        The set of output indices for the whole contraction.
    remaining : frozenset[int]
        The set of indices (corresponding to ``inputs``) of tensors still
        available to contract.
    i : int
        Index of potential tensor to contract.
    j : int
        Index of potential tensor to contract.
    size_dict dict[str, int]
        Size mapping of all the indices.

    Returns
    -------
    k12 : frozenset
        The resulting indices of the potential tensor.
    cost : int
        Estimated flop count of operation.
    """
    k1, k2 = inputs[i], inputs[j]
    either = k1 | k2
    shared = k1 & k2
    keep = frozenset.union(output, *map(inputs.__getitem__, remaining - {i, j}))

    k12 = either & keep
    cost = helpers.flop_count(either, shared - keep, 2, size_dict)

    return k12, cost


def _compute_oversize_flops(inputs, remaining, output, size_dict):
    """
    Compute the flop count for a contraction of all remaining arguments. This
    is used when a memory limit means that no pairwise contractions can be made.
    """
    idx_contraction = frozenset.union(*map(inputs.__getitem__, remaining))
    inner = idx_contraction - output
    num_terms = len(remaining)
    return helpers.flop_count(idx_contraction, inner, num_terms, size_dict)


def optimal(inputs, output, size_dict, memory_limit=None):
    """
    Computes all possible pair contractions in a depth-first recursive manner,
    sieving results based on ``memory_limit`` and the best path found so far.
    Returns the lowest cost path. This algorithm scales factoriallly with
    respect to the elements in the list ``input_sets``.

    Parameters
    ----------
    inputs : list
        List of sets that represent the lhs side of the einsum subscript.
    output : set
        Set that represents the rhs side of the overall einsum subscript.
    size_dict : dictionary
        Dictionary of index sizes.
    memory_limit : int
        The maximum number of elements in a temporary array.

    Returns
    -------
    path : list
        The optimal contraction order within the memory limit constraint.

    Examples
    --------
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set('')
    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
    >>> optimal(isets, oset, idx_sizes, 5000)
    [(0, 2), (0, 1)]
    """
    inputs = tuple(map(frozenset, inputs))
    output = frozenset(output)

    best = {'flops': float('inf'), 'ssa_path': (tuple(range(len(inputs))),)}
    size_cache = {}
    result_cache = {}

    def _optimal_iterate(path, remaining, inputs, flops):

        # reached end of path (only ever get here if flops is best found so far)
        if len(remaining) == 1:
            best['flops'] = flops
            best['ssa_path'] = path
            return

        # check all possible remaining paths
        for i, j in itertools.combinations(remaining, 2):
            if i > j:
                i, j = j, i
            key = (inputs[i], inputs[j])
            try:
                k12, flops12 = result_cache[key]
            except KeyError:
                k12, flops12 = result_cache[key] = _calc_k12_flops(inputs, output, remaining, i, j, size_dict)

            # sieve based on current best flops
            new_flops = flops + flops12
            if new_flops >= best['flops']:
                continue

            # sieve based on memory limit
            if memory_limit not in _UNLIMITED_MEM:
                try:
                    size12 = size_cache[k12]
                except KeyError:
                    size12 = size_cache[k12] = helpers.compute_size_by_dict(k12, size_dict)

                # possibly terminate this path with an all-terms einsum
                if size12 > memory_limit:
                    new_flops = flops + _compute_oversize_flops(inputs, remaining, output, size_dict)
                    if new_flops < best['flops']:
                        best['flops'] = new_flops
                        best['ssa_path'] = path + (tuple(remaining),)
                    continue

            # add contraction and recurse into all remaining
            _optimal_iterate(path=path + ((i, j),),
                             inputs=inputs + (k12,),
                             remaining=remaining - {i, j} | {len(inputs)},
                             flops=new_flops)

    _optimal_iterate(path=(),
                     inputs=inputs,
                     remaining=set(range(len(inputs))),
                     flops=0)

    return ssa_to_linear(best['ssa_path'])


# functions for comparing which of two paths is 'better'

def better_flops_first(flops, size, best_flops, best_size):
    return (flops, size) < (best_flops, best_size)


def better_size_first(flops, size, best_flops, best_size):
    return (size, flops) < (best_size, best_flops)


_BETTER_FNS = {
    'flops': better_flops_first,
    'size': better_size_first,
}


# functions for assigning a heuristic 'cost' to a potential contraction

def cost_memory_removed(size12, size1, size2, k12, k1, k2):
    """The default heuristic cost, corresponding to the total reduction in
    memory of performing a contraction.
    """
    return size12 - size1 - size2


def cost_memory_removed_jitter(size12, size1, size2, k12, k1, k2):
    """Like memory-removed, but with a slight amount of noise that breaks ties
    and thus jumbles the contractions a bit.
    """
    return random.gauss(1.0, 0.01) * (size12 - size1 - size2)


_COST_FNS = {
    'memory-removed': cost_memory_removed,
    'memory-removed-jitter': cost_memory_removed_jitter,
}


class BranchOptimizer(PathOptimizer):
    """
    Explores possible pair contractions in a depth-first recursive manner like
    the ``optimal`` approach, but with extra heuristic early pruning of branches
    as well sieving by ``memory_limit`` and the best path found so far. Returns
    the lowest cost path. This algorithm still scales factorially with respect
    to the elements in the list ``input_sets`` if ``nbranch`` is not set, but it
    scales exponentially like ``nbranch**len(input_sets)`` otherwise.

    Parameters
    ----------
    nbranch : None or int, optional
        How many branches to explore at each contraction step. If None, explore
        all possible branches. If an integer, branch into this many paths at
        each step. Defaults to None.
    cutoff_flops_factor : float, optional
        If at any point, a path is doing this much worse than the best path
        found so far was, terminate it. The larger this is made, the more paths
        will be fully explored and the slower the algorithm. Defaults to 4.
    minimize : {'flops', 'size'}, optional
        Whether to optimize the path with regard primarily to the total
        estimated flop-count, or the size of the largest intermediate. The
        option not chosen will still be used as a secondary criterion.
    cost_fn : callable, optional
        A function that returns a heuristic 'cost' of a potential contraction
        with which to sort candidates. Should have signature
        ``cost_fn(size12, size1, size2, k12, k1, k2)``.
    """

    def __init__(self, nbranch=None, cutoff_flops_factor=4, minimize='flops', cost_fn='memory-removed'):
        self.nbranch = nbranch
        self.cutoff_flops_factor = cutoff_flops_factor
        self.minimize = minimize
        self.cost_fn = _COST_FNS.get(cost_fn, cost_fn)

        self.better = _BETTER_FNS[minimize]
        self.best = {'flops': float('inf'), 'size': float('inf')}
        self.best_progress = defaultdict(lambda: float('inf'))

    @property
    def path(self):
        return ssa_to_linear(self.best['ssa_path'])

    def __call__(self, inputs, output, size_dict, memory_limit=None):
        """

        Parameters
        ----------
        input_sets : list
            List of sets that represent the lhs side of the einsum subscript
        output_set : set
            Set that represents the rhs side of the overall einsum subscript
        idx_dict : dictionary
            Dictionary of index sizes
        memory_limit : int
            The maximum number of elements in a temporary array

        Returns
        -------
        path : list
            The contraction order within the memory limit constraint.

        Examples
        --------
        >>> isets = [set('abd'), set('ac'), set('bdc')]
        >>> oset = set('')
        >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
        >>> optimal(isets, oset, idx_sizes, 5000)
        [(0, 2), (0, 1)]
        """

        inputs = tuple(map(frozenset, inputs))
        output = frozenset(output)

        size_cache = {k: helpers.compute_size_by_dict(k, size_dict) for k in inputs}
        result_cache = {}

        def _branch_iterate(path, inputs, remaining, flops, size):

            # reached end of path (only ever get here if flops is best found so far)
            if len(remaining) == 1:
                self.best['size'] = size
                self.best['flops'] = flops
                self.best['ssa_path'] = path
                return

            def _assess_candidate(k1, k2, i, j):
                # find resulting indices and flops
                try:
                    k12, flops12 = result_cache[k1, k2]
                except KeyError:
                    k12, flops12 = result_cache[k1, k2] = _calc_k12_flops(inputs, output, remaining, i, j, size_dict)

                try:
                    size12 = size_cache[k12]
                except KeyError:
                    size12 = size_cache[k12] = helpers.compute_size_by_dict(k12, size_dict)

                new_flops = flops + flops12
                new_size = max(size, size12)

                # sieve based on current best i.e. check flops and size still better
                if not self.better(new_flops, new_size, self.best['flops'], self.best['size']):
                    return None

                # compare to how the best method was doing as this point
                if new_flops < self.best_progress[len(inputs)]:
                    self.best_progress[len(inputs)] = new_flops
                # sieve based on current progress relative to best
                elif new_flops > self.cutoff_flops_factor * self.best_progress[len(inputs)]:
                    return None

                # sieve based on memory limit
                if (memory_limit not in _UNLIMITED_MEM) and (size12 > memory_limit):
                    # terminate path here, but check all-terms contract first
                    new_flops = flops + _compute_oversize_flops(inputs, remaining, output, size_dict)
                    if new_flops < self.best['flops']:
                        self.best['flops'] = new_flops
                        self.best['ssa_path'] = path + (tuple(remaining),)
                    return None

                # set cost heuristic in order to locally sort possible contractions
                size1, size2 = size_cache[inputs[i]], size_cache[inputs[j]]
                cost = self.cost_fn(size12, size1, size2, k12, k1, k2)

                return cost, flops12, new_flops, new_size, (i, j), k12

            # check all possible remaining paths
            candidates = []
            for i, j in itertools.combinations(remaining, 2):
                if i > j:
                    i, j = j, i
                k1, k2 = inputs[i], inputs[j]

                # initially ignore outer products
                if k1.isdisjoint(k2):
                    continue

                candidate = _assess_candidate(k1, k2, i, j)
                if candidate:
                    heapq.heappush(candidates, candidate)

            # assess outer products if nothing left
            if not candidates:
                for i, j in itertools.combinations(remaining, 2):
                    if i > j:
                        i, j = j, i
                    k1, k2 = inputs[i], inputs[j]
                    candidate = _assess_candidate(k1, k2, i, j)
                    if candidate:
                        heapq.heappush(candidates, candidate)

            # recurse into all or some of the best candidate contractions
            bi = 0
            while (self.nbranch is None or bi < self.nbranch) and candidates:
                _, _, new_flops, new_size, (i, j), k12 = heapq.heappop(candidates)
                _branch_iterate(path=path + ((i, j),),
                                inputs=inputs + (k12,),
                                remaining=remaining - {i, j} | {len(inputs)},
                                flops=new_flops,
                                size=new_size)
                bi += 1

        _branch_iterate(path=(),
                        inputs=inputs,
                        remaining=set(range(len(inputs))),
                        flops=0,
                        size=0)

        return self.path


def branch(inputs, output, size_dict, memory_limit=None, **optimizer_kwargs):
    optimizer = BranchOptimizer(**optimizer_kwargs)
    return optimizer(inputs, output, size_dict, memory_limit)


def _get_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2, cost_fn):
    either = k1 | k2
    two = k1 & k2
    one = either - two
    k12 = (either & output) | (two & dim_ref_counts[3]) | (one & dim_ref_counts[2])
    cost = cost_fn(helpers.compute_size_by_dict(k12, sizes), footprints[k1], footprints[k2], k12, k1, k2)
    id1 = remaining[k1]
    id2 = remaining[k2]
    if id1 > id2:
        k1, id1, k2, id2 = k2, id2, k1, id1
    cost = cost, id2, id1  # break ties to ensure determinism
    return cost, k1, k2, k12


def _push_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2s, queue, push_all, cost_fn):
    candidates = (_get_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2, cost_fn) for k2 in k2s)
    if push_all:
        # want to do this if we e.g. are using a custom 'choose_fn'
        for candidate in candidates:
            heapq.heappush(queue, candidate)
    else:
        heapq.heappush(queue, min(candidates))


def _update_ref_counts(dim_to_keys, dim_ref_counts, dims):
    for dim in dims:
        count = len(dim_to_keys[dim])
        if count <= 1:
            dim_ref_counts[2].discard(dim)
            dim_ref_counts[3].discard(dim)
        elif count == 2:
            dim_ref_counts[2].add(dim)
            dim_ref_counts[3].discard(dim)
        else:
            dim_ref_counts[2].add(dim)
            dim_ref_counts[3].add(dim)


def _simple_chooser(queue, remaining):
    """Default contraction chooser that simply takes the minimum cost option.
    """
    cost, k1, k2, k12 = heapq.heappop(queue)
    if k1 not in remaining or k2 not in remaining:
        return None  # candidate is obsolete
    return cost, k1, k2, k12


def _ssa_optimize(inputs, output, sizes, choose_fn=None, cost_fn='memory-removed'):
    """
    This is the core function for :func:`greedy` but produces a path with
    static single assignment ids rather than recycled linear ids.
    SSA ids are cheaper to work with and easier to reason about.
    """
    if len(inputs) == 1:
        # Perform a single contraction to match output shape.
        return [(0,)]

    # set the function that assigns a heuristic cost to a possible contraction
    cost_fn = _COST_FNS.get(cost_fn, cost_fn)

    # set the function that chooses which contraction to take
    if choose_fn is None:
        choose_fn = _simple_chooser
        push_all = False
    else:
        # assume chooser wants access to all possible contractions
        push_all = True

    # A dim that is common to all tensors might as well be an output dim, since it
    # cannot be contracted until the final step. This avoids an expensive all-pairs
    # comparison to search for possible contractions at each step, leading to speedup
    # in many practical problems where all tensors share a common batch dimension.
    inputs = list(map(frozenset, inputs))
    output = frozenset(output) | frozenset.intersection(*inputs)

    # Deduplicate shapes by eagerly computing Hadamard products.
    remaining = {}  # key -> ssa_id
    ssa_ids = itertools.count(len(inputs))
    ssa_path = []
    for ssa_id, key in enumerate(inputs):
        if key in remaining:
            ssa_path.append((remaining[key], ssa_id))
            remaining[key] = next(ssa_ids)
        else:
            remaining[key] = ssa_id

    # Keep track of possible contraction dims.
    dim_to_keys = defaultdict(set)
    for key in remaining:
        for dim in key - output:
            dim_to_keys[dim].add(key)

    # Keep track of the number of tensors using each dim; when the dim is no longer
    # used it can be contracted. Since we specialize to binary ops, we only care about
    # ref counts of >=2 or >=3.
    dim_ref_counts = {
        count: set(dim for dim, keys in dim_to_keys.items() if len(keys) >= count) - output
        for count in [2, 3]}

    # Compute separable part of the objective function for contractions.
    footprints = {key: helpers.compute_size_by_dict(key, sizes) for key in remaining}

    # Find initial candidate contractions.
    queue = []
    for dim, keys in dim_to_keys.items():
        keys = sorted(keys, key=remaining.__getitem__)
        for i, k1 in enumerate(keys[:-1]):
            k2s = keys[1 + i:]
            _push_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2s, queue, push_all, cost_fn)

    # Greedily contract pairs of tensors.
    while queue:

        con = choose_fn(queue, remaining)
        if con is None:
            continue  # allow choose_fn to flag all candidates obsolete
        cost, k1, k2, k12 = con

        ssa_id1 = remaining.pop(k1)
        ssa_id2 = remaining.pop(k2)
        for dim in k1 - output:
            dim_to_keys[dim].remove(k1)
        for dim in k2 - output:
            dim_to_keys[dim].remove(k2)
        ssa_path.append((ssa_id1, ssa_id2))
        if k12 in remaining:
            ssa_path.append((remaining[k12], next(ssa_ids)))
        else:
            for dim in k12 - output:
                dim_to_keys[dim].add(k12)
        remaining[k12] = next(ssa_ids)
        _update_ref_counts(dim_to_keys, dim_ref_counts, k1 | k2 - output)
        footprints[k12] = helpers.compute_size_by_dict(k12, sizes)

        # Find new candidate contractions.
        k1 = k12
        k2s = set(k2 for dim in k1 for k2 in dim_to_keys[dim])
        k2s.discard(k1)
        if k2s:
            _push_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2s, queue, push_all, cost_fn)

    # Greedily compute pairwise outer products.
    queue = [(helpers.compute_size_by_dict(key & output, sizes), ssa_id, key)
             for key, ssa_id in remaining.items()]
    heapq.heapify(queue)
    _, ssa_id1, k1 = heapq.heappop(queue)
    while queue:
        _, ssa_id2, k2 = heapq.heappop(queue)
        ssa_path.append((min(ssa_id1, ssa_id2), max(ssa_id1, ssa_id2)))
        k12 = (k1 | k2) & output
        cost = helpers.compute_size_by_dict(k12, sizes)
        ssa_id12 = next(ssa_ids)
        _, ssa_id1, k1 = heapq.heappushpop(queue, (cost, ssa_id12, k12))

    return ssa_path


def greedy(inputs, output, size_dict, memory_limit=None, cost_fn='memory-removed'):
    """
    Finds the path by a three stage algorithm:

    1. Eagerly compute Hadamard products.
    2. Greedily compute contractions to maximize ``removed_size``
    3. Greedily compute outer products.

    This algorithm scales quadratically with respect to the
    maximum number of elements sharing a common dim.

    Parameters
    ----------
    inputs : list
        List of sets that represent the lhs side of the einsum subscript
    output : set
        Set that represents the rhs side of the overall einsum subscript
    size_dict : dictionary
        Dictionary of index sizes
    memory_limit : int
        The maximum number of elements in a temporary array

    Returns
    -------
    path : list
        The contraction order (a list of tuples of ints).

    Examples
    --------
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set('')
    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
    >>> greedy(isets, oset, idx_sizes)
    [(0, 2), (0, 1)]
    """
    if memory_limit not in _UNLIMITED_MEM:
        return branch(inputs, output, size_dict, memory_limit, nbranch=1, cost_fn=cost_fn)

    ssa_path = _ssa_optimize(inputs, output, size_dict, cost_fn=cost_fn)
    return ssa_to_linear(ssa_path)


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


def _trial_ssa_path_and_cost(r, inputs, output, size_dict, choose_fn, cost_fn):
    """A single, repeatable, trial run. Returns ``ssa_path`` and cost.
    """
    if r == 0:
        # always start with the standard greedy approach
        choose_fn = None
    else:
        random.seed(r)

    ssa_path = _ssa_optimize(inputs, output, size_dict, choose_fn, cost_fn)
    cost, size = _ssa_path_compute_cost(ssa_path, inputs, output, size_dict)

    return ssa_path, cost, size


class RandomOptimizer(PathOptimizer):
    """

    Parameters
    ----------
    max_repeats : int, optional
        The maximum number of repeat trials to have.
    max_time : float, optional
        The maximum amount of time to run the algorithm for.
    minimize : {'flops', 'size'}, optional
        Whether to favour paths that minimize the total estimated flop-count or
        the size of the largest intermediate created.
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
    executor : executor-pool like, optional
        An executor-pool to optionally parallelize repeat trials over. The pool
        should have an api matching those found in the python 3 standard library
        module ``concurrent.futures``. Namely, a ``submit`` method that returns
        ``Future`` objects, themselves with ``result`` and ``cancel`` methods.
        Note that if you set ``max_repeats`` very high, that many trials will
        be submitted to the pool, resulting in a possibly significant slowdown.

    Attributes
    ----------
    path : list[tuple[int]]
        The best path found so far.
    costs : list[int]
        The list of each trial's costs found so far.
    sizes : list[int]
        The list of each trial's largest intermediate size so far.
    """

    def __init__(self, max_repeats=32, max_time=None, minimize='flops', cost_fn='memory-removed-jitter',
                 temperature=1.0, rel_temperature=True, nbranch=8, executor=None):

        if minimize not in ('flops', 'size'):
            raise ValueError("`minimize` should be one of {'flops', 'size'}.")

        self.max_repeats = max_repeats
        self.max_time = max_time
        self.minimize = minimize
        self.cost_fn = cost_fn
        self.better = _BETTER_FNS[minimize]
        self.temperature = temperature
        self.rel_temperature = rel_temperature
        self.nbranch = nbranch
        self.executor = executor

        self.costs = []
        self.sizes = []
        self.best = {'cost': float('inf'), 'size': float('inf')}

        # this keeps track of how many trials we have submitted so that (a)
        # each trial has a different seed, and (b) the very first trial can be
        # made to always be the standard greedy one
        self._r0 = 0

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

    @property
    def path(self):
        """The best path found so far.
        """
        return ssa_to_linear(self.best['ssa_path'])

    def __call__(self, inputs, output, size_dict, memory_limit):
        import time

        # start a timer?
        if self.max_time is not None:
            t0 = time.time()

        args = (inputs, output, size_dict, self.choose_fn, self.cost_fn)
        repeats = range(self._r0, self._r0 + self.max_repeats)
        self._r0 += self.max_repeats

        # create the trials lazily
        if self.executor is not None:
            # eagerly submit
            fs = [self.executor.submit(_trial_ssa_path_and_cost, r, *args) for r in repeats]
            # lazily retrieve
            trials = (f.result() for f in fs)
        else:
            trials = (_trial_ssa_path_and_cost(r, *args) for r in repeats)

        # assess the trials
        for ssa_path, cost, size in trials:

            # keep track of all costs and sizes
            self.costs.append(cost)
            self.sizes.append(size)

            # check if we have found a new best
            found_new_best = self.better(cost, size, self.best['cost'], self.best['size'])

            if found_new_best:
                self.best['cost'] = cost
                self.best['size'] = size
                self.best['ssa_path'] = ssa_path

            # check if we have run out of time
            if (self.max_time is not None) and (time.time() > t0 + self.max_time):
                # possibly cancel remaining futures
                if self.executor is not None:
                    for f in fs:
                        f.cancel()
                break

        return self.path


def random_greedy(inputs, output, idx_dict, memory_limit=None, **optimizer_kwargs):
    """
    """
    optimizer = RandomOptimizer(**optimizer_kwargs)
    return optimizer(inputs, output, idx_dict, memory_limit)
