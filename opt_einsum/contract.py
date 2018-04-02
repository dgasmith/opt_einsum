"""
Contains the primary optimization and contraction routines
"""

import numpy as np

from . import backends
from . import blas
from . import helpers
from . import parser
from . import paths


def contract_path(*operands, **kwargs):
    """
    Evaluates the lowest cost einsum-like contraction order.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.
    *operands : list of array_like
        These are the arrays for the operation.
    path : bool or list, optional (default: ``greedy``)
        Choose the type of path.

        - if a list is given uses this as the path.
        - 'greedy' An algorithm that chooses the best pair contraction
          at each step. Scales cubically with the number of terms in the
          contraction.
        - 'optimal' An algorithm that tries all possible ways of
          contracting the listed tensors. Scales exponentially with
          the number of terms in the contraction.

    use_blas : bool
        Use BLAS functions or not
    memory_limit : int, optional (default: largest input or output array size)
        Maximum number of elements allowed in intermediate arrays.

    Returns
    -------
    path : list of tuples
        The einsum path
    string_repr : str
        A printable representation of the path

    Notes
    -----
    The resulting path indicates which terms of the input contraction should be
    contracted first, the result of this contraction is then appended to the end of
    the contraction list.

    Examples
    --------

    We can begin with a chain dot example. In this case it is optimal to
    contract the b and c tensors reprsented by the first element of the path (1,
    2). The resulting tensor is added to the end of the contraction and the
    remaining contraction (0, 1) is then completed.

    >>> a = np.random.rand(2, 2)
    >>> b = np.random.rand(2, 5)
    >>> c = np.random.rand(5, 2)
    >>> path_info = opt_einsum.contract_path('ij,jk,kl->il', a, b, c)
    >>> print(path_info[0])
    [(1, 2), (0, 1)]
    >>> print(path_info[1])
      Complete contraction:  ij,jk,kl->il
             Naive scaling:  4
         Optimized scaling:  3
          Naive FLOP count:  1.600e+02
      Optimized FLOP count:  5.600e+01
       Theoretical speedup:  2.857
      Largest intermediate:  4.000e+00 elements
    -------------------------------------------------------------------------
    scaling                  current                                remaining
    -------------------------------------------------------------------------
       3                   kl,jk->jl                                ij,jl->il
       3                   jl,ij->il                                   il->il


    A more complex index transformation example.

    >>> I = np.random.rand(10, 10, 10, 10)
    >>> C = np.random.rand(10, 10)
    >>> path_info = oe.contract_path('ea,fb,abcd,gc,hd->efgh', C, C, I, C, C)

    >>> print(path_info[0])
    [(0, 2), (0, 3), (0, 2), (0, 1)]
    >>> print(path_info[1])
      Complete contraction:  ea,fb,abcd,gc,hd->efgh
             Naive scaling:  8
         Optimized scaling:  5
          Naive FLOP count:  8.000e+08
      Optimized FLOP count:  8.000e+05
       Theoretical speedup:  1000.000
      Largest intermediate:  1.000e+04 elements
    --------------------------------------------------------------------------
    scaling                  current                                remaining
    --------------------------------------------------------------------------
       5               abcd,ea->bcde                      fb,gc,hd,bcde->efgh
       5               bcde,fb->cdef                         gc,hd,cdef->efgh
       5               cdef,gc->defg                            hd,defg->efgh
       5               defg,hd->efgh                               efgh->efgh
    """

    # Make sure all keywords are valid
    valid_contract_kwargs = ['path', 'memory_limit', 'einsum_call', 'use_blas']
    unknown_kwargs = [k for (k, v) in kwargs.items() if k not in valid_contract_kwargs]
    if len(unknown_kwargs):
        raise TypeError("einsum_path: Did not understand the following kwargs: %s" % unknown_kwargs)

    path_type = kwargs.pop('path', 'greedy')
    memory_limit = kwargs.pop('memory_limit', None)

    # Hidden option, only einsum should call this
    einsum_call_arg = kwargs.pop("einsum_call", False)
    use_blas = kwargs.pop('use_blas', True)

    # Python side parsing
    input_subscripts, output_subscript, operands = parser.parse_einsum_input(operands)

    # Build a few useful list and sets
    input_list = input_subscripts.split(',')
    input_sets = [set(x) for x in input_list]
    output_set = set(output_subscript)
    indices = set(input_subscripts.replace(',', ''))

    # Get length of each unique dimension and ensure all dimensions are correct
    dimension_dict = {}
    broadcast_indices = [[] for x in range(len(input_list))]
    for tnum, term in enumerate(input_list):
        sh = operands[tnum].shape

        if len(sh) != len(term):
            raise ValueError("Einstein sum subscript %s does not contain the "
                             "correct number of indices for operand %d." % (input_subscripts[tnum], tnum))
        for cnum, char in enumerate(term):
            dim = sh[cnum]

            # Build out broadcast indices
            if dim == 1:
                broadcast_indices[tnum].append(char)

            if char in dimension_dict.keys():
                # For broadcasting cases we always want the largest dim size
                if dimension_dict[char] == 1:
                    dimension_dict[char] = dim
                elif dim not in (1, dimension_dict[char]):
                    raise ValueError("Size of label '%s' for operand %d (%d) "
                                     "does not match previous terms (%d)." % (char, tnum, dimension_dict[char], dim))
            else:
                dimension_dict[char] = dim

    # Convert broadcast inds to sets
    broadcast_indices = [set(x) for x in broadcast_indices]

    # Compute size of each input array plus the output array
    size_list = []
    for term in input_list + [output_subscript]:
        size_list.append(helpers.compute_size_by_dict(term, dimension_dict))
    out_size = max(size_list)

    if memory_limit is None:
        memory_arg = out_size
    else:
        if memory_limit < 1:
            if memory_limit == -1:
                memory_arg = int(1e20)
            else:
                raise ValueError("Memory limit must be larger than 0, or -1")
        else:
            memory_arg = int(memory_limit)

    # Compute naive cost
    # This isnt quite right, need to look into exactly how einsum does this
    # indices_in_input = input_subscripts.replace(',', '')
    # inne
    inner_product = (sum(len(x) for x in input_sets) - len(indices)) > 0
    naive_cost = helpers.flop_count(indices, inner_product, len(input_list), dimension_dict)

    # Compute the path
    if not isinstance(path_type, str):
        path = path_type
    elif len(input_list) == 1:
        # Nothing to be optimized
        path = [(0, )]
    elif len(input_list) == 2:
        # Nothing to be optimized
        path = [(0, 1)]
    elif indices == output_set:
        # If no rank reduction leave it to einsum
        path = [tuple(range(len(input_list)))]
    elif path_type in ["greedy", "opportunistic"]:
        path = paths.greedy(input_sets, output_set, dimension_dict, memory_arg)
    elif path_type == "optimal":
        path = paths.optimal(input_sets, output_set, dimension_dict, memory_arg)
    else:
        raise KeyError("Path name %s not found" % path_type)

    cost_list = []
    scale_list = []
    size_list = []
    contraction_list = []

    # Build contraction tuple (positions, gemm, einsum_str, remaining)
    for cnum, contract_inds in enumerate(path):
        # Make sure we remove inds from right to left
        contract_inds = tuple(sorted(list(contract_inds), reverse=True))

        contract_tuple = helpers.find_contraction(contract_inds, input_sets, output_set)
        out_inds, input_sets, idx_removed, idx_contract = contract_tuple

        # Compute cost, scale, and size
        cost = helpers.flop_count(idx_contract, idx_removed, len(contract_inds), dimension_dict)
        cost_list.append(cost)
        scale_list.append(len(idx_contract))
        size_list.append(helpers.compute_size_by_dict(out_inds, dimension_dict))

        bcast = set()
        tmp_inputs = []
        for x in contract_inds:
            tmp_inputs.append(input_list.pop(x))
            bcast |= broadcast_indices.pop(x)

        new_bcast_inds = bcast - idx_removed

        # If were broadcasting, nix blas
        if use_blas and not len(idx_removed & bcast):
            do_blas = blas.can_blas(tmp_inputs, out_inds, idx_removed)
        else:
            do_blas = False

        # Last contraction
        if (cnum - len(path)) == -1:
            idx_result = output_subscript
        else:
            sort_result = [(dimension_dict[ind], ind) for ind in out_inds]
            idx_result = "".join([x[1] for x in sorted(sort_result)])

        input_list.append(idx_result)
        broadcast_indices.append(new_bcast_inds)
        einsum_str = ",".join(tmp_inputs) + "->" + idx_result

        contraction = (contract_inds, idx_removed, einsum_str, input_list[:], do_blas)
        contraction_list.append(contraction)

    opt_cost = sum(cost_list)

    if einsum_call_arg:
        return operands, contraction_list

    # Return the path along with a nice string representation
    overall_contraction = input_subscripts + "->" + output_subscript
    header = ("scaling", "BLAS", "current", "remaining")

    path_print = "  Complete contraction:  %s\n" % overall_contraction
    path_print += "         Naive scaling:  %d\n" % len(indices)
    path_print += "     Optimized scaling:  %d\n" % max(scale_list)
    path_print += "      Naive FLOP count:  %.3e\n" % naive_cost
    path_print += "  Optimized FLOP count:  %.3e\n" % opt_cost
    path_print += "   Theoretical speedup:  %3.3f\n" % (naive_cost / float(opt_cost))
    path_print += "  Largest intermediate:  %.3e elements\n" % max(size_list)
    path_print += "-" * 80 + "\n"
    path_print += "%6s %11s %22s %37s\n" % header
    path_print += "-" * 80

    for n, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str, remaining, do_blas = contraction
        remaining_str = ",".join(remaining) + "->" + output_subscript
        path_run = (scale_list[n], do_blas, einsum_str, remaining_str)
        path_print += "\n%4d %14s %22s %37s" % path_run

    return path, path_print


def _einsum(*operands, **kwargs):
    """Base einsum, but with pre-parse for valid characters if string given.
    """
    fn = backends.get_func('einsum', kwargs.pop('backend', 'numpy'))

    if not isinstance(operands[0], str):
        return fn(*operands, **kwargs)

    einsum_str, operands = operands[0], operands[1:]

    # Do we need to temporarily map indices into [a-z,A-Z] range?
    if not parser.has_valid_einsum_chars_only(einsum_str):

        # Explicitly find output str first so as to maintain order
        if '->' not in einsum_str:
            einsum_str += '->' + parser.find_output_str(einsum_str)

        einsum_str = parser.convert_to_valid_einsum_chars(einsum_str)

    return fn(einsum_str, *operands, **kwargs)


def _transpose(x, axes, backend='numpy'):
    """Base transpose.
    """
    try:
        return x.transpose(axes)
    except AttributeError:
        # some libraries don't implement method version
        fn = backends.get_func('transpose', backend)
        return fn(x, axes)


def _tensordot(x, y, axes, backend='numpy'):
    """Base tensordot.
    """
    fn = backends.get_func('tensordot', backend)
    return fn(x, y, axes=axes)


# Rewrite einsum to handle different cases
def contract(*operands, **kwargs):
    """
    contract(subscripts, *operands, out=None, dtype=None, order='K', casting='safe',
             use_blas=True, optimize=True, memory_limit=None, backend='numpy')

    Evaluates the Einstein summation convention on the operands. A drop in
    replacment for NumPy's einsum function that optimizes the order of contraction
    to reduce overall scaling at the cost of several intermediate arrays.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.
    *operands : list of array_like
        These are the arrays for the operation.
    out : array_like
        A output array in which set the resulting output.
    dtype : str
        The dtype of the given contraction, see np.einsum.
    order : str
        The order of the resulting contraction, see np.einsum.
    casting : str
        The casting procedure for operations of different dtype, see np.einsum.
    use_blas : bool
        Do you use BLAS for valid operations, may use extra memory for more intermediates.
    optimize : bool, str, or list, optional (default: ``greedy``)
        Choose the type of path.

        - if a list is given uses this as the path.
        - 'greedy' An algorithm that chooses the best pair contraction
          at each step. Scales cubically with the number of terms in the
          contraction.
        - 'optimal' An algorithm that tries all possible ways of
          contracting the listed tensors. Scales exponentially with
          the number of terms in the contraction.

    memory_limit : int or None (default : None)
        The upper limit of the size of tensor created, by default this will be
        Give the upper bound of the largest intermediate tensor contract will build.
        By default (None) will size the ``memory_limit`` as the largest input tensor.
        Users can also specify ``-1`` to allow arbitrarily large tensors to be built.
    backend : str, optional (default: ``numpy``)
        Which library to use to perform the required ``tensordot``, ``transpose``
        and ``einsum`` calls. Should match the types of arrays supplied, See
        :func:`contract_expression` for generating expressions which convert
        numpy arrays to and from the backend library automatically.

    Returns
    -------
    out : array_like
        The result of the einsum expression.

    Notes
    -----
    This function should produce result identical to that of NumPy's einsum
    function. The primary difference is `contract` will attempt to form
    intermediates which reduce the overall scaling of the given einsum contraction.
    By default the worst intermediate formed will be equal to that of the largest
    input array. For large einsum expressions with many input arrays this can
    provide arbitrarily large (1000 fold+) speed improvements.

    For contractions with just two tensors this function will attempt to use
    NumPy's built in BLAS functionality to ensure that the given operation is
    preformed in an optimal manner. When NumPy is linked to a threaded BLAS, potenital
    speedsups are on the order of 20-100 for a six core machine.

    Examples
    --------

    See :func:`opt_einsum.contract_path` or :func:`numpy.einsum`

    """
    optimize_arg = kwargs.pop('optimize', True)
    if optimize_arg is True:
        optimize_arg = 'greedy'

    valid_einsum_kwargs = ['out', 'dtype', 'order', 'casting']
    einsum_kwargs = {k: v for (k, v) in kwargs.items() if k in valid_einsum_kwargs}

    # If no optimization, run pure einsum
    if optimize_arg is False:
        return _einsum(*operands, **einsum_kwargs)

    # Grab non-einsum kwargs
    use_blas = kwargs.pop('use_blas', True)
    memory_limit = kwargs.pop('memory_limit', None)
    backend = kwargs.pop('backend', 'numpy')
    gen_expression = kwargs.pop('gen_expression', False)

    # Make sure remaining keywords are valid for einsum
    unknown_kwargs = [k for (k, v) in kwargs.items() if k not in valid_einsum_kwargs]
    if len(unknown_kwargs):
        raise TypeError("Did not understand the following kwargs: %s" % unknown_kwargs)

    if gen_expression:
        full_str = operands[0]

    # Build the contraction list and operand
    operands, contraction_list = contract_path(
        *operands, path=optimize_arg, memory_limit=memory_limit, einsum_call=True, use_blas=use_blas)

    # check if performing contraction or just building expression
    if gen_expression:
        return ContractExpression(full_str, contraction_list, **einsum_kwargs)

    return _core_contract(operands, contraction_list, backend=backend, **einsum_kwargs)


def _core_contract(operands, contraction_list, backend='numpy', **einsum_kwargs):
    """Inner loop used to perform an actual contraction given the output
    from a ``contract_path(..., einsum_call=True)`` call.
    """

    # Special handeling if out is specified
    out_array = einsum_kwargs.pop('out', None)
    specified_out = out_array is not None

    # try and do as much as possible without einsum if not available
    no_einsum = not backends.has_einsum(backend)

    # Start contraction loop
    for num, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str, remaining, blas_flag = contraction
        tmp_operands = []
        for x in inds:
            tmp_operands.append(operands.pop(x))

        # Do we need to deal with the output?
        handle_out = specified_out and ((num + 1) == len(contraction_list))

        # Call tensordot (check if should prefer einsum, but only if available)
        if blas_flag and ('EINSUM' not in blas_flag or no_einsum):

            # Checks have already been handled
            input_str, results_index = einsum_str.split('->')
            input_left, input_right = input_str.split(',')

            tensor_result = input_left + input_right
            for s in idx_rm:
                tensor_result = tensor_result.replace(s, "")

            # Find indices to contract over
            left_pos, right_pos = [], []
            for s in idx_rm:
                left_pos.append(input_left.find(s))
                right_pos.append(input_right.find(s))

            # Contract!
            new_view = _tensordot(*tmp_operands, axes=(tuple(left_pos), tuple(right_pos)), backend=backend)

            # Build a new view if needed
            if (tensor_result != results_index) or handle_out:

                transpose = tuple(map(tensor_result.index, results_index))
                new_view = _transpose(new_view, axes=transpose, backend=backend)

                if handle_out:
                    out_array[:] = new_view

        # Call einsum
        else:
            # If out was specified
            if handle_out:
                einsum_kwargs["out"] = out_array

            # Do the contraction
            new_view = _einsum(einsum_str, *tmp_operands, backend=backend, **einsum_kwargs)

        # Append new items and derefernce what we can
        operands.append(new_view)
        del tmp_operands, new_view

    if specified_out:
        return out_array
    else:
        return operands[0]


class ContractExpression:
    """Helper class for storing an explicit ``contraction_list`` which can
    then be repeatedly called solely with the array arguments.
    """

    def __init__(self, contraction, contraction_list, **einsum_kwargs):
        self.contraction = contraction
        self.contraction_list = contraction_list
        self.einsum_kwargs = einsum_kwargs
        self.num_args = len(contraction.split('->')[0].split(','))

    def _normal_contract(self, arrays, out=None, backend='numpy'):
        """The normal, core contraction.
        """
        return _core_contract(list(arrays), self.contraction_list, out=out,
                              backend=backend, **self.einsum_kwargs)

    def _convert_contract(self, arrays, out, backend):
        """Special contraction, i.e. contraction with a different backend
        but converting to and from that backend. Checks for
        ``self._{backend}_contract``, generates it if is missing, then calls it
        with ``arrays``.
        """
        convert_fn = "_{}_contract".format(backend)

        if not hasattr(self, convert_fn):
            setattr(self, convert_fn, backends.build_expression(backend, arrays, self))

        result = getattr(self, convert_fn)(*arrays)

        if out is not None:
            out[()] = result
            return out

        return result

    def __call__(self, *arrays, **kwargs):
        """Evaluate this expression with a set of arrays.

        Parameters
        ----------
        arrays : seq of array
            The arrays to supply as input to the expression.
        out : array, optional (default: ``None``)
            If specified, output the result into this array.
        backend : str, optional  (default: ``numpy``)
            Perform the contraction with this backend library. If numpy arrays
            are supplied then try to convert them to and from the correct
            backend array type.
        """

        if len(arrays) != self.num_args:
            raise ValueError("This `ContractExpression` takes exactly %s array arguments "
                             "but received %s." % (self.num_args, len(arrays)))

        backend = kwargs.pop('backend', 'numpy')
        out = kwargs.pop('out', None)
        if kwargs:
            raise ValueError("The only valid keyword arguments to a `ContractExpression` "
                             "call are `out=` or `backend=`. Got: %s." % kwargs)

        try:
            # Check if the backend requires special preparation / calling
            #   but also ignore non-numpy arrays -> assume user wants same type back
            if backend in backends.CONVERT_BACKENDS and isinstance(arrays[0], np.ndarray):
                return self._convert_contract(arrays, out, backend)

            return self._normal_contract(arrays, out, backend)

        except ValueError as err:
            original_msg = str(err.args) if err.args else ""
            msg = ("Internal error while evaluating `ContractExpression`. Note that few checks are performed"
                   " - the number and rank of the array arguments must match the original expression. "
                   "The internal error was: '%s'" % original_msg, )
            err.args = msg
            raise

    def __repr__(self):
        return "ContractExpression('%s')" % self.contraction

    def __str__(self):
        s = "<ContractExpression> for '%s':" % self.contraction
        for i, c in enumerate(self.contraction_list):
            s += "\n  %i.  " % (i + 1)
            s += "'%s'" % c[2] + (" [%s]" % c[-1] if c[-1] else "")
        if self.einsum_kwargs:
            s += "\neinsum_kwargs=%s" % self.einsum_kwargs
        return s


class _ShapeOnly(np.ndarray):
    """Dummy ``numpy.ndarray`` which has a shape only - for generating
    contract expressions.
    """

    def __init__(self, shape):
        self.shape = shape


def contract_expression(subscripts, *shapes, **kwargs):
    """Generate an reusable expression for a given contraction with
    specific shapes, which can for example be cached.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.
    shapes : sequence of integer tuples
        Shapes of the arrays to optimize the contraction for.
    kwargs :
        Passed on to ``contract_path`` or ``einsum``. See ``contract``.

    Returns
    -------
    expr : ContractExpression
        Callable with signature ``expr(*arrays, out=None, backend='numpy')``
        where the array's shapes should match ``shapes``.

    Notes
    -----
    - The `out` keyword argument should be supplied to the generated expression
      rather than this function.
    - The `backend` keyword argument should also be supplied to the generated
      expression. If numpy arrays are supplied, if possible they will be
      converted to and back from the correct backend array type.
    - The generated expression will work with any arrays which have
      the same rank (number of dimensions) as the original shapes, however, if
      the actual sizes are different, the expression may no longer be optimal.

    Examples
    --------

    >>> expr = contract_expression("ab,bc->ac", (3, 4), (4, 5))
    >>> a, b = np.random.rand(3, 4), np.random.rand(4, 5)
    >>> c = expr(a, b)
    >>> np.allclose(c, a @ b)
    True

    """
    if not kwargs.get('optimize', True):
        raise ValueError("Can only generate expressions for optimized contractions.")

    for arg in ('out', 'backend'):
        if kwargs.get(arg, None) is not None:
            raise ValueError("'%s' should only be specified when calling a "
                             "`ContractExpression`, not when building it." % arg)

    dummy_arrays = [_ShapeOnly(s) for s in shapes]

    return contract(subscripts, *dummy_arrays, gen_expression=True, **kwargs)
