"""
Required functions for optimized contractions of numpy arrays using jax.
"""

from __future__ import absolute_import

import numpy as np


__all__ = ["build_expression", "evaluate_constants"]


def build_expression(_, expr):  # pragma: no cover
    """Build a jax function based on ``arrays`` and ``expr``.
    """
    import jax

    jax_expr = jax.jit(expr._contract)

    def jax_contract(*arrays):
        return np.asarray(jax_expr(arrays))

    return jax_contract


def evaluate_constants(const_arrays, expr):  # pragma: no cover
    """Convert constant arguments to jax arrays, and perform any possible
    constant contractions.
    """
    return expr(*const_arrays, backend='jax', evaluate_constants=True)
