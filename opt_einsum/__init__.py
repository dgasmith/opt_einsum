"""
Main init function for opt_einsum.
"""

from opt_einsum import blas, helpers, path_random, paths
from opt_einsum.contract import contract, contract_expression, contract_path
from opt_einsum.parser import get_symbol
from opt_einsum.path_random import RandomGreedy
from opt_einsum.paths import BranchBound, DynamicProgramming
from opt_einsum.sharing import shared_intermediates

# Handle versioneer
from opt_einsum._version import get_versions  # isort:skip

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions

paths.register_path_fn("random-greedy", path_random.random_greedy)
paths.register_path_fn("random-greedy-128", path_random.random_greedy_128)
