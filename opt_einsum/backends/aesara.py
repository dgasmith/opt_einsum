"""
Required functions for optimized contractions of numpy arrays using aesara.
"""

import numpy as np

from ..sharing import to_backend_cache_wrap

__all__ = ["to_aesara", "build_expression", "evaluate_constants"]


@to_backend_cache_wrap(constants=True)
def to_aesara(array, constant=False):
    """Convert a numpy array to ``aesara.tensor.TensorType`` instance."""
    import aesara.tensor as at

    if isinstance(array, np.ndarray):
        if constant:
            return at.constant(array)

        return at.TensorType(dtype=array.dtype, broadcastable=[False] * len(array.shape))()

    return array


def build_expression(arrays, expr):
    """Build a aesara function based on ``arrays`` and ``expr``."""
    import aesara

    in_vars = [to_aesara(array) for array in arrays]
    out_var = expr._contract(in_vars, backend="aesara")

    # don't supply constants to graph
    graph_ins = [x for x in in_vars if not isinstance(x, aesara.tensor.TensorConstant)]
    graph = aesara.function(graph_ins, out_var)

    def aesara_contract(*arrays):
        return graph(*[x for x in arrays if not isinstance(x, aesara.tensor.TensorConstant)])

    return aesara_contract


def evaluate_constants(const_arrays, expr):
    # compute the partial graph of new inputs
    const_arrays = [to_aesara(x, constant=True) for x in const_arrays]
    new_ops, new_contraction_list = expr(*const_arrays, backend="aesara", evaluate_constants=True)

    # evaluate the new inputs and convert to aesara shared tensors
    new_ops = [None if x is None else to_aesara(x.eval(), constant=True) for x in new_ops]

    return new_ops, new_contraction_list
