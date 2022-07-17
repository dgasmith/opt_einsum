"""
Required functions for optimized contractions of arrays using array API-compliant backends.
"""
import sys
from importlib.metadata import entry_points
from typing import Callable
from types import ModuleType

import numpy as np

from ..sharing import to_backend_cache_wrap


def discover_array_api_eps():
    """Discover array API backends and return their entry points."""
    if sys.version_info >= (3, 10):
        return entry_points(group='array_api')
    else:
        # Deprecated - will raise warning in Python versions >= 3.10
        return entry_points().get('array_api', [])

def make_to_array(array_api: ModuleType) -> Callable:
    """Make a ``to_[array_api]`` function for the given array API."""
    @to_backend_cache_wrap
    def to_array(array):  # pragma: no cover
        if isinstance(array, np.ndarray):
            return array_api.asarray(array)
        return array
    return to_array

def make_build_expression(array_api: ModuleType) -> Callable:
    """Make a ``build_expression`` function for the given array API."""
    def build_expression(_, expr):  # pragma: no cover
        """Build an array API function based on ``arrays`` and ``expr``."""
        def array_api_contract(*arrays):
            return expr._contract([make_to_array(array_api)(x) for x in arrays], backend=array_api.__name__)
        return array_api_contract
    return build_expression

def make_evaluate_constants(array_api: ModuleType) -> Callable:
    def evaluate_constants(const_arrays, expr):  # pragma: no cover
        """Convert constant arguments to cupy arrays, and perform any possible constant contractions.
        """
        return expr(*[make_to_array(array_api)(x) for x in const_arrays], backend="cupy", evaluate_constants=True)
    return evaluate_constants

to_array_api = {}
build_expression = {}
evaluate_constants = {}

for ep in discover_array_api_eps():
    _array_api = ep.load()
    to_array_api[ep.value] = make_to_array(_array_api)
    build_expression[ep.value] = make_build_expression(_array_api)
    evaluate_constants[ep.value] = make_evaluate_constants(_array_api)

__all__ = [
    'discover_array_api_eps',
    "to_array_api",
    "build_expression",
    "evaluate_constants"
]