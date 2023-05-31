# The _version.py file is managed by setuptools-scm
#   and is not in version control.
from osqp._version import version as __version__  # noqa: F401
from osqp.interface import (  # noqa: F401
    OSQP,
    constant,
    algebra_available,
    algebras_available,
    default_algebra,
)
