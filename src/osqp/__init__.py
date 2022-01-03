# The _version.py file is managed by setuptools-scm
#   and is not in version control.
from ._version import version as __version__

from osqp.interface import OSQP, constant, algebra_available, algebras_available, default_algebra