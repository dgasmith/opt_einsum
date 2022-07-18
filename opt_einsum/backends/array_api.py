"""
Required functions for optimized contractions of arrays using array API-compliant backends.
"""
import sys
from typing import Callable
from types import ModuleType

import numpy as np

from ..sharing import to_backend_cache_wrap


def discover_array_apis():
    """Discover array API backends."""
    if sys.version_info >= (3, 8):
        from importlib.metadata import entry_points

        if sys.version_info >= (3, 10):
            eps = entry_points(group="array_api")
        else:
            # Deprecated - will raise warning in Python versions >= 3.10
            eps = entry_points().get("array_api", [])
        return [ep.load() for ep in eps]
    else:
        # importlib.metadata was introduced in Python 3.8, so it isn't available here. Unable to discover any array APIs.
        return []


def make_to_array_function(array_api: ModuleType) -> Callable:
    """Make a ``to_[array_api]`` function for the given array API."""

    @to_backend_cache_wrap
    def to_array(array):  # pragma: no cover
        if isinstance(array, np.ndarray):
            return array_api.asarray(array)
        return array

    return to_array


def make_build_expression_function(array_api: ModuleType) -> Callable:
    """Make a ``build_expression`` function for the given array API."""

    def build_expression(_, expr):  # pragma: no cover
        """Build an array API function based on ``arrays`` and ``expr``."""

        def array_api_contract(*arrays):
            return expr._contract([make_to_array_function(array_api)(x) for x in arrays], backend=array_api.__name__)

        return array_api_contract

    return build_expression


def make_evaluate_constants_function(array_api: ModuleType) -> Callable:
    def evaluate_constants(const_arrays, expr):  # pragma: no cover
        """Convert constant arguments to cupy arrays, and perform any possible constant contractions."""
        return expr(
            *[make_to_array_function(array_api)(x) for x in const_arrays],
            backend=array_api.__name__,
            evaluate_constants=True,
        )

    return evaluate_constants


_array_apis = discover_array_apis()
to_array_api = {api.__name__: make_to_array_function(api) for api in _array_apis}
build_expression = {api.__name__: make_build_expression_function(api) for api in _array_apis}
evaluate_constants = {api.__name__: make_evaluate_constants_function(api) for api in _array_apis}

__all__ = ["discover_array_apis", "to_array_api", "build_expression", "evaluate_constants"]
