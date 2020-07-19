"""
Contains the primary optimization and contraction routines.
"""

from collections import namedtuple
from decimal import Decimal

import numpy as np

from . import backends, blas, helpers, parser, paths, sharing

__all__ = ["contract_path", "contract", "format_const_einsum_str", "ContractExpression", "shape_only"]


class PathInfo(object):
    """A printable object to contain information about a contraction path.

    Attributes
    ----------
    naive_cost : int
        The estimate FLOP cost of a naive einsum contraction.
    opt_cost : int
        The estimate FLOP cost of this optimized contraction path.
    largest_intermediate : int
        The number of elements in the largest intermediate array that will be
        produced during the contraction.
    """
    def __init__(self, contraction_list, input_subscripts, output_subscript, indices, path, scale_list, naive_cost,
                 opt_cost, size_list, size_dict):
        self.contraction_list = contraction_list
        self.input_subscripts = input_subscripts
        self.output_subscript = output_subscript
        self.path = path
        self.indices = indices
        self.scale_list = scale_list
        self.naive_cost = Decimal(naive_cost)
        self.opt_cost = Decimal(opt_cost)
        self.speedup = self.naive_cost / self.opt_cost
        self.size_list = size_list
        self.size_dict = size_dict

        self.shapes = [tuple(size_dict[k] for k in ks) for ks in input_subscripts.split(',')]
        self.eq = "{}->{}".format(input_subscripts, output_subscript)
        self.largest_intermediate = Decimal(max(size_list))

    def __repr__(self):
        # Return the path along with a nice string representation
        header = ("scaling", "BLAS", "current", "remaining")

        path_print = [
            "  Complete contraction:  {}\n".format(self.eq), "         Naive scaling:  {}\n".format(len(self.indices)),
            "     Optimized scaling:  {}\n".format(max(self.scale_list)), "      Naive FLOP count:  {:.3e}\n".format(
                self.naive_cost), "  Optimized FLOP count:  {:.3e}\n".format(self.opt_cost),
            "   Theoretical speedup:  {:.3e}\n".format(self.speedup),
            "  Largest intermediate:  {:.3e} elements\n".format(self.largest_intermediate), "-" * 80 + "\n",
            "{:>6} {:>11} {:>22} {:>37}\n".format(*header), "-" * 80
        ]

        for n, contraction in enumerate(self.contraction_list):
            inds, idx_rm, einsum_str, remaining, do_blas = contraction

            if remaining is not None:
                remaining_str = ",".join(remaining) + "->" + self.output_subscript
            else:
                remaining_str = "..."
            size_remaining = max(0, 56 - max(22, len(einsum_str)))

            path_run = (self.scale_list[n], do_blas, einsum_str, remaining_str, size_remaining)
            path_print.append("\n{:>4} {:>14} {:>22}    {:>{}}".format(*path_run))

        return "".join(path_print)


def _choose_memory_arg(memory_limit, size_list):
    if memory_limit == 'max_input':
        return max(size_list)

    if memory_limit is None:
        return None

    if memory_limit < 1:
        if memory_limit == -1:
            return None
        else:
            raise ValueError("Memory limit must be larger than 0, or -1")

    return int(memory_limit)


_VALID_CONTRACT_KWARGS = {'optimize', 'path', 'memory_limit', 'einsum_call', 'use_blas', 'shapes'}


def contract_path(*operands, **kwargs):
    """
    Find a contraction order 'path', without performing the contraction.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.
    *operands : list of array_like
        These are the arrays for the operation.
    optimize : str, list or bool, optional (default: ``auto``)
        Choose the type of path.

        - if a list is given uses this as the path.
        - ``'optimal'`` An algorithm that explores all possible ways of
          contracting the listed tensors. Scales factorially with the number of
          terms in the contraction.
        - ``'branch-all'`` An algorithm like optimal but that restricts itself
          to searching 'likely' paths. Still scales factorially.
        - ``'branch-2'`` An even more restricted version of 'branch-all' that
          only searches the best two options at each step. Scales exponentially
          with the number of terms in the contraction.
        - ``'greedy'`` An algorithm that heuristically chooses the best pair
          contraction at each step.
        - ``'auto'`` Choose the best of the above algorithms whilst aiming to
          keep the path finding time below 1ms.

    use_blas : bool
        Use BLAS functions or not
    memory_limit : int, optional (default: None)
        Maximum number of elements allowed in intermediate arrays.
    shapes : bool, optional
        Whether ``contract_path`` should assume arrays (the default) or array
        shapes have been supplied.

    Returns
    -------
    path : list of tuples
        The einsum path
    PathInfo : str
        A printable object containing various information about the path found.

    Notes
    -----
    The resulting path indicates which terms of the input contraction should be
    contracted first, the result of this contraction is then appended to the end of
    the contraction list.

    Examples
    --------

    We can begin with a chain dot example. In this case, it is optimal to
    contract the b and c tensors represented by the first element of the path (1,
    2). The resulting tensor is added to the end of the contraction and the
    remaining contraction, ``(0, 1)``, is then executed.

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
    unknown_kwargs = set(kwargs) - _VALID_CONTRACT_KWARGS
    if len(unknown_kwargs):
        raise TypeError("einsum_path: Did not understand the following kwargs: {}".format(unknown_kwargs))

    path_type = kwargs.pop('optimize', 'auto')

    memory_limit = kwargs.pop('memory_limit', None)
    shapes = kwargs.pop('shapes', False)

    # Hidden option, only einsum should call this
    einsum_call_arg = kwargs.pop("einsum_call", False)
    use_blas = kwargs.pop('use_blas', True)

    # Python side parsing
    input_subscripts, output_subscript, operands = parser.parse_einsum_input(operands)

    # Build a few useful list and sets
    input_list = input_subscripts.split(',')
    input_sets = [set(x) for x in input_list]
    if shapes:
        input_shps = operands
    else:
        input_shps = [x.shape for x in operands]
    output_set = set(output_subscript)
    indices = set(input_subscripts.replace(',', ''))

    # Get length of each unique dimension and ensure all dimensions are correct
    size_dict = {}
    for tnum, term in enumerate(input_list):
        sh = input_shps[tnum]

        if len(sh) != len(term):
            raise ValueError("Einstein sum subscript '{}' does not contain the "
                             "correct number of indices for operand {}.".format(input_list[tnum], tnum))
        for cnum, char in enumerate(term):
            dim = int(sh[cnum])

            if char in size_dict:
                # For broadcasting cases we always want the largest dim size
                if size_dict[char] == 1:
                    size_dict[char] = dim
                elif dim not in (1, size_dict[char]):
                    raise ValueError("Size of label '{}' for operand {} ({}) does not match previous "
                                     "terms ({}).".format(char, tnum, size_dict[char], dim))
            else:
                size_dict[char] = dim

    # Compute size of each input array plus the output array
    size_list = [helpers.compute_size_by_dict(term, size_dict) for term in input_list + [output_subscript]]
    memory_arg = _choose_memory_arg(memory_limit, size_list)

    num_ops = len(input_list)

    # Compute naive cost
    # This isnt quite right, need to look into exactly how einsum does this
    # indices_in_input = input_subscripts.replace(',', '')

    inner_product = (sum(len(x) for x in input_sets) - len(indices)) > 0
    naive_cost = helpers.flop_count(indices, inner_product, num_ops, size_dict)

    # Compute the path
    if not isinstance(path_type, (str, paths.PathOptimizer)):
        # Custom path supplied
        path = path_type
    elif num_ops <= 2:
        # Nothing to be optimized
        path = [tuple(range(num_ops))]
    elif isinstance(path_type, paths.PathOptimizer):
        # Custom path optimizer supplied
        path = path_type(input_sets, output_set, size_dict, memory_arg)
    else:
        path_optimizer = paths.get_path_fn(path_type)
        path = path_optimizer(input_sets, output_set, size_dict, memory_arg)

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
        cost = helpers.flop_count(idx_contract, idx_removed, len(contract_inds), size_dict)
        cost_list.append(cost)
        scale_list.append(len(idx_contract))
        size_list.append(helpers.compute_size_by_dict(out_inds, size_dict))

        tmp_inputs = [input_list.pop(x) for x in contract_inds]
        tmp_shapes = [input_shps.pop(x) for x in contract_inds]

        if use_blas:
            do_blas = blas.can_blas(tmp_inputs, out_inds, idx_removed, tmp_shapes)
        else:
            do_blas = False

        # Last contraction
        if (cnum - len(path)) == -1:
            idx_result = output_subscript
        else:
            # use tensordot order to minimize transpositions
            all_input_inds = "".join(tmp_inputs)
            idx_result = "".join(sorted(out_inds, key=all_input_inds.find))

        shp_result = parser.find_output_shape(tmp_inputs, tmp_shapes, idx_result)

        input_list.append(idx_result)
        input_shps.append(shp_result)

        einsum_str = ",".join(tmp_inputs) + "->" + idx_result

        # for large expressions saving the remaining terms at each step can
        # incur a large memory footprint - and also be messy to print
        if len(input_list) <= 20:
            remaining = tuple(input_list)
        else:
            remaining = None

        contraction = (contract_inds, idx_removed, einsum_str, remaining, do_blas)
        contraction_list.append(contraction)

    opt_cost = sum(cost_list)

    if einsum_call_arg:
        return operands, contraction_list

    path_print = PathInfo(contraction_list, input_subscripts, output_subscript, indices, path, scale_list, naive_cost,
                          opt_cost, size_list, size_dict)

    return path, path_print


@sharing.einsum_cache_wrap
def _einsum(*operands, **kwargs):
    """Base einsum, but with pre-parse for valid characters if a string is given.
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


def _default_transpose(x, axes):
    #  most libraries implement a method version
    return x.transpose(axes)


@sharing.transpose_cache_wrap
def _transpose(x, axes, backend='numpy'):
    """Base transpose.
    """
    fn = backends.get_func('transpose', backend, _default_transpose)
    return fn(x, axes)


@sharing.tensordot_cache_wrap
def _tensordot(x, y, axes, backend='numpy'):
    """Base tensordot.
    """
    fn = backends.get_func('tensordot', backend)
    return fn(x, y, axes=axes)


# Rewrite einsum to handle different cases
def contract(*operands, **kwargs):
    """
    contract(subscripts, *operands, out=None, dtype=None, order='K', casting='safe', use_blas=True, optimize=True, memory_limit=None, backend='numpy')

    Evaluates the Einstein summation convention on the operands. A drop in
    replacement for NumPy's einsum function that optimizes the order of contraction
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
    optimize : str, list or bool, optional (default: ``auto``)
        Choose the type of path.

        - if a list is given uses this as the path.
        - ``'optimal'`` An algorithm that explores all possible ways of
          contracting the listed tensors. Scales factorially with the number of
          terms in the contraction.
        - ``'dp'`` A faster (but essentially optimal) algorithm that uses
          dynamic programming to exhaustively search all contraction paths
          without outer-products.
        - ``'greedy'`` An cheap algorithm that heuristically chooses the best
          pairwise contraction at each step. Scales linearly in the number of
          terms in the contraction.
        - ``'random-greedy'`` Run a randomized version of the greedy algorithm
          32 times and pick the best path.
        - ``'random-greedy-128'`` Run a randomized version of the greedy
          algorithm 128 times and pick the best path.
        - ``'branch-all'`` An algorithm like optimal but that restricts itself
          to searching 'likely' paths. Still scales factorially.
        - ``'branch-2'`` An even more restricted version of 'branch-all' that
          only searches the best two options at each step. Scales exponentially
          with the number of terms in the contraction.
        - ``'auto'`` Choose the best of the above algorithms whilst aiming to
          keep the path finding time below 1ms.
        - ``'auto-hq'`` Aim for a high quality contraction, choosing the best
          of the above algorithms whilst aiming to keep the path finding time
          below 1sec.

    memory_limit : {None, int, 'max_input'} (default: None)
        Give the upper bound of the largest intermediate tensor contract will build.

        - None or -1 means there is no limit
        - 'max_input' means the limit is set as largest input tensor
        - a positive integer is taken as an explicit limit on the number of elements

        The default is None. Note that imposing a limit can make contractions
        exponentially slower to perform.
    backend : str, optional (default: ``auto``)
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
    This function should produce a result identical to that of NumPy's einsum
    function. The primary difference is ``contract`` will attempt to form
    intermediates which reduce the overall scaling of the given einsum contraction.
    By default the worst intermediate formed will be equal to that of the largest
    input array. For large einsum expressions with many input arrays this can
    provide arbitrarily large (1000 fold+) speed improvements.

    For contractions with just two tensors this function will attempt to use
    NumPy's built-in BLAS functionality to ensure that the given operation is
    preformed optimally. When NumPy is linked to a threaded BLAS, potential
    speedups are on the order of 20-100 for a six core machine.

    Examples
    --------

    See :func:`opt_einsum.contract_path` or :func:`numpy.einsum`

    """
    optimize_arg = kwargs.pop('optimize', True)
    if optimize_arg is True:
        optimize_arg = 'auto'

    valid_einsum_kwargs = ['out', 'dtype', 'order', 'casting']
    einsum_kwargs = {k: v for (k, v) in kwargs.items() if k in valid_einsum_kwargs}

    # If no optimization, run pure einsum
    if optimize_arg is False:
        return _einsum(*operands, **einsum_kwargs)

    # Grab non-einsum kwargs
    use_blas = kwargs.pop('use_blas', True)
    memory_limit = kwargs.pop('memory_limit', None)
    backend = kwargs.pop('backend', 'auto')
    gen_expression = kwargs.pop('_gen_expression', False)
    constants_dict = kwargs.pop('_constants_dict', {})

    # Make sure remaining keywords are valid for einsum
    unknown_kwargs = [k for (k, v) in kwargs.items() if k not in valid_einsum_kwargs]
    if len(unknown_kwargs):
        raise TypeError("Did not understand the following kwargs: {}".format(unknown_kwargs))

    if gen_expression:
        full_str = operands[0]

    # Build the contraction list and operand
    operands, contraction_list = contract_path(*operands,
                                               optimize=optimize_arg,
                                               memory_limit=memory_limit,
                                               einsum_call=True,
                                               use_blas=use_blas)

    # check if performing contraction or just building expression
    if gen_expression:
        return ContractExpression(full_str, contraction_list, constants_dict, **einsum_kwargs)

    return _core_contract(operands, contraction_list, backend=backend, **einsum_kwargs)


def infer_backend(x):
    return x.__class__.__module__.split('.')[0]


def parse_backend(arrays, backend):
    """Find out what backend we should use, dipatching based on the first
    array if ``backend='auto'`` is specified.
    """
    if backend != 'auto':
        return backend
    backend = infer_backend(arrays[0])

    # some arrays will be defined in modules that don't implement tensordot
    # etc. so instead default to numpy
    if not backends.has_tensordot(backend):
        return 'numpy'

    return backend


def _core_contract(operands, contraction_list, backend='auto', evaluate_constants=False, **einsum_kwargs):
    """Inner loop used to perform an actual contraction given the output
    from a ``contract_path(..., einsum_call=True)`` call.
    """

    # Special handling if out is specified
    out_array = einsum_kwargs.pop('out', None)
    specified_out = out_array is not None
    backend = parse_backend(operands, backend)

    # try and do as much as possible without einsum if not available
    no_einsum = not backends.has_einsum(backend)

    # Start contraction loop
    for num, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str, _, blas_flag = contraction

        # check if we are performing the pre-pass of an expression with constants,
        #     if so, break out upon finding first non-constant (None) operand
        if evaluate_constants and any(operands[x] is None for x in inds):
            return operands, contraction_list[num:]

        tmp_operands = [operands.pop(x) for x in inds]

        # Do we need to deal with the output?
        handle_out = specified_out and ((num + 1) == len(contraction_list))

        # Call tensordot (check if should prefer einsum, but only if available)
        if blas_flag and ('EINSUM' not in blas_flag or no_einsum):

            # Checks have already been handled
            input_str, results_index = einsum_str.split('->')
            input_left, input_right = input_str.split(',')

            tensor_result = "".join(s for s in input_left + input_right if s not in idx_rm)

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

        # Append new items and dereference what we can
        operands.append(new_view)
        del tmp_operands, new_view

    if specified_out:
        return out_array
    else:
        return operands[0]


def format_const_einsum_str(einsum_str, constants):
    """Add brackets to the constant terms in ``einsum_str``. For example:

        >>> format_const_einsum_str('ab,bc,cd->ad', [0, 2])
        'bc,[ab,cd]->ad'

    No-op if there are no constants.
    """
    if not constants:
        return einsum_str

    if "->" in einsum_str:
        lhs, rhs = einsum_str.split('->')
        arrow = "->"
    else:
        lhs, rhs, arrow = einsum_str, "", ""

    wrapped_terms = ["[{}]".format(t) if i in constants else t for i, t in enumerate(lhs.split(','))]

    formatted_einsum_str = "{}{}{}".format(','.join(wrapped_terms), arrow, rhs)

    # merge adjacent constants
    formatted_einsum_str = formatted_einsum_str.replace("],[", ',')
    return formatted_einsum_str


class ContractExpression:
    """Helper class for storing an explicit ``contraction_list`` which can
    then be repeatedly called solely with the array arguments.
    """
    def __init__(self, contraction, contraction_list, constants_dict, **einsum_kwargs):
        self.contraction_list = contraction_list
        self.einsum_kwargs = einsum_kwargs
        self.contraction = format_const_einsum_str(contraction, constants_dict.keys())

        # need to know _full_num_args to parse constants with, and num_args to call with
        self._full_num_args = contraction.count(',') + 1
        self.num_args = self._full_num_args - len(constants_dict)

        # likewise need to know full contraction list
        self._full_contraction_list = contraction_list

        self._constants_dict = constants_dict
        self._evaluated_constants = {}
        self._backend_expressions = {}

    def evaluate_constants(self, backend='auto'):
        """Convert any constant operands to the correct backend form, and
        perform as many contractions as possible to create a new list of
        operands, stored in ``self._evaluated_constants[backend]``. This also
        makes sure ``self.contraction_list`` only contains the remaining,
        non-const operations.
        """
        # prepare a list of operands, with `None` for non-consts
        tmp_const_ops = [self._constants_dict.get(i, None) for i in range(self._full_num_args)]
        backend = parse_backend(tmp_const_ops, backend)

        # get the new list of operands with constant operations performed, and remaining contractions
        try:
            new_ops, new_contraction_list = backends.evaluate_constants(backend, tmp_const_ops, self)
        except KeyError:
            new_ops, new_contraction_list = self(*tmp_const_ops, backend=backend, evaluate_constants=True)

        self._evaluated_constants[backend] = new_ops
        self.contraction_list = new_contraction_list

    def _get_evaluated_constants(self, backend):
        """Retrieve or generate the cached list of constant operators (mixed
        in with None representing non-consts) and the remaining contraction
        list.
        """
        try:
            return self._evaluated_constants[backend]
        except KeyError:
            self.evaluate_constants(backend)
            return self._evaluated_constants[backend]

    def _get_backend_expression(self, arrays, backend):
        try:
            return self._backend_expressions[backend]
        except KeyError:
            fn = backends.build_expression(backend, arrays, self)
            self._backend_expressions[backend] = fn
            return fn

    def _contract(self, arrays, out=None, backend='auto', evaluate_constants=False):
        """The normal, core contraction.
        """
        contraction_list = self._full_contraction_list if evaluate_constants else self.contraction_list

        return _core_contract(list(arrays),
                              contraction_list,
                              out=out,
                              backend=backend,
                              evaluate_constants=evaluate_constants,
                              **self.einsum_kwargs)

    def _contract_with_conversion(self, arrays, out, backend, evaluate_constants=False):
        """Special contraction, i.e., contraction with a different backend
        but converting to and from that backend. Retrieves or generates a
        cached expression using ``arrays`` as templates, then calls it
        with ``arrays``.

        If ``evaluate_constants=True``, perform a partial contraction that
        prepares the constant tensors and operations with the right backend.
        """
        # convert consts to correct type & find reduced contraction list
        if evaluate_constants:
            return backends.evaluate_constants(backend, arrays, self)

        result = self._get_backend_expression(arrays, backend)(*arrays)

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
        out = kwargs.pop('out', None)
        backend = kwargs.pop('backend', 'auto')
        backend = parse_backend(arrays, backend)
        evaluate_constants = kwargs.pop('evaluate_constants', False)

        if kwargs:
            raise ValueError("The only valid keyword arguments to a `ContractExpression` "
                             "call are `out=` or `backend=`. Got: {}.".format(kwargs))

        correct_num_args = self._full_num_args if evaluate_constants else self.num_args

        if len(arrays) != correct_num_args:
            raise ValueError("This `ContractExpression` takes exactly {} array arguments "
                             "but received {}.".format(self.num_args, len(arrays)))

        if self._constants_dict and not evaluate_constants:
            # fill in the missing non-constant terms with newly supplied arrays
            ops_var, ops_const = iter(arrays), self._get_evaluated_constants(backend)
            ops = [next(ops_var) if op is None else op for op in ops_const]
        else:
            ops = arrays

        try:
            # Check if the backend requires special preparation / calling
            #   but also ignore non-numpy arrays -> assume user wants same type back
            if backends.has_backend(backend) and all(isinstance(x, np.ndarray) for x in arrays):
                return self._contract_with_conversion(ops, out, backend, evaluate_constants=evaluate_constants)

            return self._contract(ops, out, backend, evaluate_constants=evaluate_constants)

        except ValueError as err:
            original_msg = str(err.args) if err.args else ""
            msg = ("Internal error while evaluating `ContractExpression`. Note that few checks are performed"
                   " - the number and rank of the array arguments must match the original expression. "
                   "The internal error was: '{}'".format(original_msg), )
            err.args = msg
            raise

    def __repr__(self):
        if self._constants_dict:
            constants_repr = ", constants={}".format(sorted(self._constants_dict))
        else:
            constants_repr = ""
        return "<ContractExpression('{}'{})>".format(self.contraction, constants_repr)

    def __str__(self):
        s = [self.__repr__()]
        for i, c in enumerate(self.contraction_list):
            s.append("\n  {}.  ".format(i + 1))
            s.append("'{}'".format(c[2]) + (" [{}]".format(c[-1]) if c[-1] else ""))
        if self.einsum_kwargs:
            s.append("\neinsum_kwargs={}".format(self.einsum_kwargs))
        return "".join(s)


Shaped = namedtuple('Shaped', ['shape'])


def shape_only(shape):
    """Dummy ``numpy.ndarray`` which has a shape only - for generating
    contract expressions.
    """
    return Shaped(shape)


def contract_expression(subscripts, *shapes, **kwargs):
    """Generate a reusable expression for a given contraction with
    specific shapes, which can, for example, be cached.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.
    shapes : sequence of integer tuples
        Shapes of the arrays to optimize the contraction for.
    constants : sequence of int, optional
        The indices of any constant arguments in ``shapes``, in which case the
        actual array should be supplied at that position rather than just a
        shape. If these are specified, then constant parts of the contraction
        between calls will be reused. Additionally, if a GPU-enabled backend is
        used for example, then the constant tensors will be kept on the GPU,
        minimizing transfers.
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
    - Constant operations will be computed upon the first call with a particular
      backend, then subsequently reused.

    Examples
    --------

    Basic usage:

        >>> expr = contract_expression("ab,bc->ac", (3, 4), (4, 5))
        >>> a, b = np.random.rand(3, 4), np.random.rand(4, 5)
        >>> c = expr(a, b)
        >>> np.allclose(c, a @ b)
        True

    Supply ``a`` as a constant:

        >>> expr = contract_expression("ab,bc->ac", a, (4, 5), constants=[0])
        >>> expr
        <ContractExpression('[ab],bc->ac', constants=[0])>

        >>> c = expr(b)
        >>> np.allclose(c, a @ b)
        True

    """
    if not kwargs.get('optimize', True):
        raise ValueError("Can only generate expressions for optimized contractions.")

    for arg in ('out', 'backend'):
        if kwargs.get(arg, None) is not None:
            raise ValueError("'{}' should only be specified when calling a "
                             "`ContractExpression`, not when building it.".format(arg))

    if not isinstance(subscripts, str):
        subscripts, shapes = parser.convert_interleaved_input((subscripts, ) + shapes)

    kwargs['_gen_expression'] = True

    # build dict of constant indices mapped to arrays
    constants = kwargs.pop('constants', ())
    constants_dict = {i: shapes[i] for i in constants}
    kwargs['_constants_dict'] = constants_dict

    # apart from constant arguments, make dummy arrays
    dummy_arrays = [s if i in constants else shape_only(s) for i, s in enumerate(shapes)]

    return contract(subscripts, *dummy_arrays, **kwargs)
