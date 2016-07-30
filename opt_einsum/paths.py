# All possible opt_einsum paths live here


def compute_size_by_dict(indices, idx_dict):
    """
    Computes the product of the elements in indices based on the dictionary
    idx_dict.

    Parameters
    ----------
    indices : iterable
        Indices to base the product on.
    idx_dict : dictionary
        Dictionary of index sizes

    Returns
    -------
    ret : int
        The resulting product.

    Examples
    --------
    >>> compute_size_by_dict('abbc', {'a': 2, 'b':3, 'c':5})
    90

    """
    ret = 1
    for i in indices:
        ret *= idx_dict[i]
    return ret


def find_contraction(positions, input_sets, output_set):
    """
    Finds the contraction for a given set of input and output sets.

    Paramaters
    ----------
    positions : iterable
        Integer positions of terms used in the contraction.
    input_sets : list
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript

    Returns
    -------
    new_result : set
        The indices of the resulting contraction
    remaining : list
        List of sets that have not been contracted
    idx_removed : set
        Indices removed from the entire contraction
    idx_contraction : set
        The indices used in the current contraction

    Examples
    --------
    >>> p = (0, 2)
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set('ac')
    >>> find_contraction(p, isets, oset)
    ({'a', 'c'}, [{'a', 'c'}, {'a', 'c'}], {'b', 'd'}, {'a', 'b', 'c', 'd'})
    """

    idx_contract = set()
    idx_remain = output_set.copy()
    remaining = []
    for ind, value in enumerate(input_sets):
        if ind in positions:
            idx_contract |= value
        else:
            remaining.append(value)
            idx_remain |= value

    new_result = idx_remain & idx_contract
    idx_removed = (idx_contract - new_result)
    remaining.append(new_result)
    return (new_result, remaining, idx_removed, idx_contract)


def optimal(input_sets, output_set, idx_dict, memory_limit):
    """
    Computes all possible pair contractions, sieves the results based
    on ``memory_limit`` and returns the lowest cost path.

    Paramaters
    ----------
    input_sets : list
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript
    idx_dict : dictionary
        Dictionary of index sizes
    memory_limit : int
        The maximum number of elements in a temporary array.

    Returns
    -------
    path : list
        The optimal order of pair contractions.

    Examples
    --------
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set('')
    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
    >>> _path_optimal(isets, oset, idx_sizes, 5000)
    [(0, 2), (0, 1)]
    """


    new = [] # Incase input_sets < 2
    current = [(0, [], input_sets)]
    for iteration in range(len(input_sets) - 1):
        new = []

        # Grab all unique pairs
        comb_iter = []
        for x in range(len(input_sets) - iteration):
            for y in range(x + 1, len(input_sets) - iteration):
                comb_iter.append((x, y))

        for curr in current:
            cost, positions, remaining = curr
            for con in comb_iter:

                contract = find_contraction(con, remaining, output_set)
                new_result, new_input_sets, idx_removed, idx_contract = contract

                # Sieve the results based on memory_limit
                new_size = compute_size_by_dict(new_result, idx_dict)
                if new_size > memory_limit:
                    continue

                # Find cost
                new_cost = compute_size_by_dict(idx_contract, idx_dict)
                if idx_removed:
                    new_cost *= 2

                # Build (total_cost, positions, indices_remaining)
                new_cost += cost
                new_pos = positions + [con]
                new.append((new_cost, new_pos, new_input_sets))

        # Update list to iterate over
        current = new

    # If we have not found anything return single einsum contraction
    if len(new) == 0:
        return [tuple(range(len(input_sets)))]

    path = min(new, key=lambda x: x[0])[1]
    return path


def greedy(input_sets, output_set, idx_dict, memory_limit):
    """
    Finds the best pair contraction at each iteration. The best pair is found
    by minimizing the tuple ``(-prod(indices_removed), cost)``.  Another way to say
    this is it tries to remove the largest index at the lowest cost.  What this
    amounts to is prioritizing matrix multiplication or inner product operations,
    then Hadamard like operations, and finally outer operations. Outer products are
    limited by ``memory_limit``.

    Paramaters
    ----------
    input_sets : list
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript
    idx_dict : dictionary
        Dictionary of index sizes
    memory_limit_limit : int
        The maximum number of elements in a temporary array.

    Returns
    -------
    path : list
        The greedy order of pair contractions.

    Examples
    --------
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set('')
    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
    >>> _path_greedy(isets, oset, idx_sizes, 5000)
    [(0, 2), (0, 1)]
    """

    path = []
    for iteration in range(len(input_sets) - 1):
        iteration_results = []
        comb_iter = []

        # Grab all unique pairs
        for x in range(len(input_sets)):
            for y in range(x + 1, len(input_sets)):
                comb_iter.append((x, y))

        for positions in comb_iter:

            contract = find_contraction(positions, input_sets, output_set)
            idx_result, new_input_sets, idx_removed, idx_contract = contract

            # Sieve the results based on memory_limit
            if compute_size_by_dict(idx_result, idx_dict) > memory_limit:
                continue

            # Build sort tuple
            removed_size = compute_size_by_dict(idx_removed, idx_dict)
            cost = compute_size_by_dict(idx_contract, idx_dict)
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
