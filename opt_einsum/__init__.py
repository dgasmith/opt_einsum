"""
Main init function for opt_einsum.
"""

from . import blas, helpers, path_random, paths
from .contract import contract, contract_expression, contract_path
from .parser import get_symbol
from .path_random import RandomGreedy
from .paths import BranchBound, DynamicProgramming
from .sharing import shared_intermediates

# Handle versioneer
from ._version import get_versions  # isort:skip

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions

paths.register_path_fn("random-greedy", path_random.random_greedy)
paths.register_path_fn("random-greedy-128", path_random.random_greedy_128)
