"""
Contains the path technology behind opt_einsum in addition to several path helpers
"""
from __future__ import absolute_import, division, print_function

import heapq
import itertools
from collections import defaultdict

import numpy as np

from . import helpers

__all__ = ["optimal", "greedy", "cheap"]


def optimal(input_sets, output_set, idx_dict, memory_limit):
    """
    Computes all possible pair contractions, sieves the results based
    on ``memory_limit`` and returns the lowest cost path. This algorithm
    scales factorial with respect to the elements in the list ``input_sets``.

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
        The optimal contraction order within the memory limit constraint.

    Examples
    --------
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set('')
    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
    >>> optimal(isets, oset, idx_sizes, 5000)
    [(0, 2), (0, 1)]
    """

    full_results = [(0, [], input_sets)]
    for iteration in range(len(input_sets) - 1):
        iter_results = []

        # Compute all unique pairs
        comb_iter = tuple(itertools.combinations(range(len(input_sets) - iteration), 2))

        for curr in full_results:
            cost, positions, remaining = curr
            for con in comb_iter:

                # Find the contraction
                contract = helpers.find_contraction(con, remaining, output_set)
                new_result, new_input_sets, idx_removed, idx_contract = contract

                # Sieve the results based on memory_limit
                new_size = helpers.compute_size_by_dict(new_result, idx_dict)
                if new_size > memory_limit:
                    continue

                # Build (total_cost, positions, indices_remaining)
                total_cost = cost + helpers.flop_count(idx_contract, idx_removed, len(con), idx_dict)
                new_pos = positions + [con]
                iter_results.append((total_cost, new_pos, new_input_sets))

        # Update combinatorial list, if we did not find anything return best
        # path + remaining contractions
        if not iter_results:
            path = min(full_results, key=lambda x: x[0])[1]
            path += [tuple(range(len(input_sets) - iteration))]
            return path

        # Update list to iterate over
        full_results = iter_results

    # If we have not found anything return single einsum contraction
    if len(full_results) == 0:
        return [tuple(range(len(input_sets)))]

    path = min(full_results, key=lambda x: x[0])[1]
    return path


def _parse_possible_contraction(positions, input_sets, output_set, idx_dict, memory_limit, path_cost, naive_cost):
    """Compute the cost (removed size + flops) and resultant indices for
    performing the contraction specified by ``positions``.

    Parameters
    ----------
    positions : tuple of int
        The locations of the proposed tensors to contract.
    input_sets : list of sets
        The indices found on each tensors.
    output_set : set
        The output indices of the expression.
    idx_dict : dict
        Mapping of each index to its size.
    memory_limit : int
        The total allowed size for an intermediary tensor.
    path_cost : int
        The contraction cost so far.
    naive_cost : int
        The cost of the unoptimized expression.

    Returns
    -------
    cost : (int, int)
        A tuple containing the size of any indices removed, and the flop cost.
    positions : tuple of int
        The locations of the proposed tensors to contract.
    new_input_sets : list of sets
        The resulting new list of indices if this proposed contraction is performed.

    """

    # Find the contraction
    contract = helpers.find_contraction(positions, input_sets, output_set)
    idx_result, new_input_sets, idx_removed, idx_contract = contract

    # Sieve the results based on memory_limit
    new_size = helpers.compute_size_by_dict(idx_result, idx_dict)
    if new_size > memory_limit:
        return None

    # Build sort tuple
    old_sizes = (helpers.compute_size_by_dict(input_sets[p], idx_dict) for p in positions)
    removed_size = sum(old_sizes) - new_size

    # NB: removed_size used to be just the size of any removed indices i.e.:
    #     helpers.compute_size_by_dict(idx_removed, idx_dict)
    cost = helpers.flop_count(idx_contract, idx_removed, len(positions), idx_dict)
    sort = (-removed_size, cost)

    # Sieve based on total cost as well
    if (path_cost + cost) > naive_cost:
        return None

    # Add contraction to possible choices
    return [sort, positions, new_input_sets]


def _update_other_results(results, best):
    """Update the positions and provisional input_sets of ``results`` based on
    performing the contraction result ``best``. Remove any involving the tensors
    contracted.

    Parameters
    ----------
    results :
        List of contraction results produced by ``_parse_possible_contraction``.
    best :
        The best contraction of ``results`` i.e. the one that will be performed.

    Returns
    -------
    mod_results :
        The list of modifed results, updated with outcome of ``best`` contraction.
    """

    best_con = best[1]
    bx, by = best_con
    mod_results = []

    for cost, (x, y), con_sets in results:

        # Ignore results involving tensors just contracted
        if x in best_con or y in best_con:
            continue

        # Update the input_sets
        del con_sets[by - int(by > x) - int(by > y)]
        del con_sets[bx - int(bx > x) - int(bx > y)]
        con_sets.insert(-1, best[2][-1])

        # Update the position indices
        mod_con = x - int(x > bx) - int(x > by), y - int(y > bx) - int(y > by)
        mod_results.append((cost, mod_con, con_sets))

    return mod_results


def greedy(input_sets, output_set, idx_dict, memory_limit):
    """
    Finds the path by contracting the best pair until the input list is
    exhausted. The best pair is found by minimizing the tuple
    ``(-removed_size, cost)``.  What this amounts to is prioritizing
    inner product operations, matrix multiplication, then Hadamard like
    operations, and finally outer operations. Outer products are limited by
    ``memory_limit`` and are ignored until no other operations are
    available. This algorithm scales quadratically with respect to the
    number of elements in the list ``input_sets``.

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
        The greedy contraction order within the memory limit constraint.

    Examples
    --------
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set('')
    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
    >>> greedy(isets, oset, idx_sizes, 5000)
    [(0, 2), (0, 1)]
    """

    # Build up a naive cost
    contract = helpers.find_contraction(range(len(input_sets)), input_sets, output_set)
    idx_result, new_input_sets, idx_removed, idx_contract = contract
    naive_cost = helpers.flop_count(idx_contract, idx_removed, len(input_sets), idx_dict)

    comb_iter = itertools.combinations(range(len(input_sets)), 2)
    iteration_results = []

    path_cost = 0
    path = []

    for iteration in range(len(input_sets) - 1):

        # Iterate over all pairs on first step, only previously found pairs on subsequent steps
        for positions in comb_iter:

            # Always initially ignore outer products
            if input_sets[positions[0]].isdisjoint(input_sets[positions[1]]):
                continue

            result = _parse_possible_contraction(positions, input_sets, output_set, idx_dict, memory_limit, path_cost,
                                                 naive_cost)
            if result is not None:
                iteration_results.append(result)

        # If we do not have a inner contraction, rescan pairs including outer products
        if len(iteration_results) == 0:

            # Then check the outer products
            for positions in itertools.combinations(range(len(input_sets)), 2):
                result = _parse_possible_contraction(positions, input_sets, output_set, idx_dict, memory_limit,
                                                     path_cost, naive_cost)
                if result is not None:
                    iteration_results.append(result)

            # If we still did not find any remaining contractions, default back to einsum like behavior
            if len(iteration_results) == 0:
                path.append(tuple(range(len(input_sets))))
                break

        # Sort based on first index
        best = min(iteration_results, key=lambda x: x[0])

        # Now propagate as many unused contractions as possible to next iteration
        iteration_results = _update_other_results(iteration_results, best)

        # Next iteration only compute contractions with the new tensor
        # All other contractions have been accounted for
        input_sets = best[2]
        new_tensor_pos = len(input_sets) - 1
        comb_iter = ((i, new_tensor_pos) for i in range(new_tensor_pos))

        # Update path and total cost
        path.append(best[1])
        path_cost += best[0][1]

    return path


def ssa_to_linear(ssa_path):
    """
    Convert a path with static single assignment ids to a path with recycled
    linear ids.
    """
    ids = np.arange(sum(map(len, ssa_path)), dtype=np.int64)
    path = []
    for i, ssa_ids in enumerate(ssa_path):
        path.append(tuple(ids[ssa_id] for ssa_id in ssa_ids))
        for ssa_id in ssa_ids:
            ids[ssa_id:] -= 1
    return path


def linear_to_ssa(path):
    """
    Convert a path with recycled linear ids to a path with static single
    assignment ids.
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


def _get_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2):
    either = k1 | k2
    two = k1 & k2
    one = either - two
    k12 = (either & output) | (two & dim_ref_counts[3]) | (one & dim_ref_counts[2])
    footprint12 = helpers.compute_size_by_dict(k12, sizes)
    cost = footprint12 - footprints[k1] - footprints[k2]
    id1 = remaining[k1]
    id2 = remaining[k2]
    cost = cost, min(id1, id2), max(id1, id2)  # break ties to ensure determinism
    return cost, k1, k2, k12


def _push_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2s, queue):
    if not k2s:
        return
    candidate = min(_get_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2)
                    for k2 in k2s)
    heapq.heappush(queue, candidate)


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


def _ssa_optimize(inputs, output, sizes):
    """
    This has an interface similar to :func:`optimize` but produces a path with
    static single assignment ids rather than recycled linear ids.
    SSA ids are cheaper to work with and easier to reason about.
    """
    # A dim common to all terms might as well be an output dim.
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

    # Compute footprints of each tensor.
    footprints = {key: helpers.compute_size_by_dict(key, sizes)
                  for key in remaining}

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

    # Find initial candidate contractions.
    queue = []
    for dim, keys in dim_to_keys.items():
        keys = list(keys)
        for i, k1 in enumerate(keys):
            k2s = keys[:i]
            _push_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2s, queue)

    # Greedily contract pairs of tensors.
    while queue:
        cost, k1, k2, k12 = heapq.heappop(queue)
        if k1 not in remaining or k2 not in remaining:
            continue  # candidate is obsolete

        ssa_id1 = remaining.pop(k1)
        ssa_id2 = remaining.pop(k2)
        for dim in k1 - output:
            dim_to_keys[dim].remove(k1)
        for dim in k2 - output:
            dim_to_keys[dim].remove(k2)
        ssa_path.append((ssa_id2, ssa_id1))
        if k12 in remaining:
            ssa_path.append((remaining[k12], next(ssa_ids)))
        else:
            footprints[k12] = helpers.compute_size_by_dict(k12, sizes)
            for dim in k12 - output:
                dim_to_keys[dim].add(k12)
        remaining[k12] = next(ssa_ids)
        _update_ref_counts(dim_to_keys, dim_ref_counts, k1 | k2 - output)

        # Find new candidate contractions.
        k1 = k12
        k2s = set(k2 for dim in k1 for k2 in dim_to_keys[dim] if k2 != k1)
        _push_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2s, queue)

    # Greedily compute pairwise outer products.
    queue = [(len(key & output), ssa_id, key) for key, ssa_id in remaining.items()]
    heapq.heapify(queue)
    _, ssa_id1, k1 = heapq.heappop(queue)
    while queue:
        _, ssa_id2, k2 = heapq.heappop(queue)
        ssa_path.append((ssa_id1, ssa_id2))
        k12 = (k1 | k2) & output
        cost = len(k12)
        ssa_id12 = next(ssa_ids)
        _, ssa_id1, k1 = heapq.heappushpop(queue, (cost, ssa_id12, k12))

    # Perform one final reduction to match output shape.
    ssa_path.append((ssa_id1,))
    return ssa_path


def cheap(inputs, output, idx_dict):
    """
    Produces an optimization path similar to the greedy strategy
    :func:`opt_einsum.paths.greedy`. This optimizer is cheaper and less
    accurate than the default ``opt_einsum`` optimizer.

    :param list inputs: A list of input shapes. These can be strings or sets or
        frozensets of characters.
    :param str output: An output shape. This can be a string or set or
        frozenset of characters.
    :param dict sizes: A mapping from dimensions (characters in inputs) to ints
        that are the sizes of those dimensions.
    :return: An optimization path: a list if tuples of contraction indices.
    rtype: list
    """

    """
    Finds the path by a quick-and-dirty method.

    1. Eagerly compute Hadamard products.
    2. Greedily compute contractions to maximize ``removed_size``
    3. Greedily compute outer products.
    4. Finally call einsum with a single arg to fix output shape.

    This algorithm scales quadratically with respect to the
    maximum number of elements sharing a common dim.

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
        The greedy contraction order within the memory limit constraint.

    Examples
    --------
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set('')
    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
    >>> greedy(isets, oset, idx_sizes, 5000)
    [(0, 2), (0, 1)]
    """
    ssa_path = _ssa_optimize(inputs, output, idx_dict)
    return ssa_to_linear(ssa_path)
