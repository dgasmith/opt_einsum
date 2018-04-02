"""
Main init function for opt_einsum.
"""

from .contract import contract, contract_path, contract_expression
from . import paths
from . import blas
from . import helpers

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
