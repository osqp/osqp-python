import os
from warnings import warn
import functools
import importlib
import numpy as np


_ALGEBRAS = ('cuda', 'mkl', 'default')   # Highest->Lowest priority of algebras that are tried in turn
# Mapping from algebra to loadable module
_ALGEBRA_MODULES = {
    'cuda': 'osqp_cuda',
    'mkl' : 'osqp_mkl',
    'default': 'osqp.ext_default'
}
OSQP_ALGEBRA = os.environ.get('OSQP_ALGEBRA')      # If envvar is set, that algebra is used by default


@functools.lru_cache(maxsize=4)
def algebra_available(algebra):
    assert algebra in _ALGEBRAS, f'Unknown algebra {algebra}'
    module = _ALGEBRA_MODULES[algebra]

    try:
        importlib.import_module(module)
    except ImportError:
        return False
    else:
        return True


@functools.lru_cache(maxsize=1)
def algebras_available():
    return [algebra for algebra in _ALGEBRAS if algebra_available(algebra)]


@functools.lru_cache(maxsize=1)
def default_algebra():
    if OSQP_ALGEBRA is not None:
        return OSQP_ALGEBRA
    for algebra in _ALGEBRAS:
        if algebra_available(algebra):
            return algebra
    raise RuntimeError('No algebra backend available!')


def constant(which, algebra=None):
    algebra = algebra or default_algebra()
    m = importlib.import_module(_ALGEBRA_MODULES[algebra])
    _constant = getattr(m, which, None)

    # If the constant was exported directly as an atomic type in the extension, use it;
    # Otherwise it's an enum out of which we can obtain the raw value
    if isinstance(_constant, (int, float, str)):
        return _constant
    elif _constant is not None:
        return _constant.value
    else:
        # Handle special cases
        if which == 'OSQP_NAN':
            return np.nan

        solvers = ('QDLDL_SOLVER', 'MKL_PARDISO_SOLVER', 'CUDA_PCG_SOLVER')
        if which in solvers:
            warn(f"The constant {which} is provided only for backward compatibility."
                 "Please use OSQP_ALGEBRA directly.")
            return solvers.index(which)
        raise RuntimeError(f"Unknown constant {which}")


class OSQP:
    def __new__(cls, *args, **kwargs):
        algebra = kwargs.pop('algebra', OSQP_ALGEBRA) or default_algebra()
        from .new_interface import OSQP as OSQP_pybind11
        return OSQP_pybind11(*args, **kwargs, algebra=algebra)