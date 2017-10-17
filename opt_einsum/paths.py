"""
Contains the path technology behind opt_einsum in addition to several path helpers
"""

from . import helpers


def optimal(input_sets, output_set, idx_dict, memory_limit):
    """
    Computes all possible pair contractions, sieves the results based
    on ``memory_limit`` and returns the lowest cost path. This algorithm
    scales factorial with respect to the elements in the list ``input_sets``.

    Paramaters
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
    >>> _path_optimal(isets, oset, idx_sizes, 5000)
    [(0, 2), (0, 1)]
    """

    full_results = [(0, [], input_sets)]
    for iteration in range(len(input_sets) - 1):
        iter_results = []

        # Compute all unique pairs
        comb_iter = []
        for x in range(len(input_sets) - iteration):
            for y in range(x + 1, len(input_sets) - iteration):
                comb_iter.append((x, y))

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

                # Find cost
                new_cost = helpers.compute_size_by_dict(idx_contract, idx_dict)
                if idx_removed:
                    new_cost *= 2

                # Build (total_cost, positions, indices_remaining)
                new_cost += cost
                new_pos = positions + [con]
                iter_results.append((new_cost, new_pos, new_input_sets))

        # Update combinatorial list, if we did not find anything return best
        # path + remaining contractions
        if iter_results:
            full_results = iter_results
        else:
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


def greedy(input_sets, output_set, idx_dict, memory_limit):
    """
    Finds the path by contracting the best pair until the input list is
    exhausted. The best pair is found by minimizing the tuple
    ``(-prod(indices_removed), cost)``.  What this amounts to is prioritizing
    matrix multiplication or inner product operations, then Hadamard like
    operations, and finally outer operations. Outer products are limited by
    ``memory_limit``. This algorithm scales cubically with respect to the number of
    elements in the list ``input_sets``.

    Paramaters
    ----------
    input_sets : list
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript
    idx_dict : dictionary
        Dictionary of index sizes
    memory_limit_limit : int
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
    >>> _path_greedy(isets, oset, idx_sizes, 5000)
    [(0, 2), (0, 1)]
    """

    if len(input_sets) == 1:
        return [(0, )]

    path = []
    for iteration in range(len(input_sets) - 1):
        iteration_results = []
        comb_iter = []

        # Compute all unique pairs
        for x in range(len(input_sets)):
            for y in range(x + 1, len(input_sets)):
                comb_iter.append((x, y))

        for positions in comb_iter:

            # Find the contraction
            contract = helpers.find_contraction(positions, input_sets, output_set)
            idx_result, new_input_sets, idx_removed, idx_contract = contract

            # Sieve the results based on memory_limit
            if helpers.compute_size_by_dict(idx_result, idx_dict) > memory_limit:
                continue

            # Build sort tuple
            removed_size = helpers.compute_size_by_dict(idx_removed, idx_dict)
            cost = helpers.compute_size_by_dict(idx_contract, idx_dict)
            sort = (-removed_size, cost)

            # Add contraction to possible choices
            iteration_results.append([sort, positions, new_input_sets])

        # If we did not find a new contraction contract remaining
        if len(iteration_results) == 0:
            path.append(tuple(range(len(input_sets))))
            break

        # Sort based on first index
        best = min(iteration_results, key=lambda x: x[0])
        path.append(best[1])
        input_sets = best[2]

    return path
