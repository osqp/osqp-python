import sys
import os
from types import SimpleNamespace
from enum import IntEnum
import shutil
import subprocess
import warnings
import importlib
import importlib.resources
import numpy as np
import scipy.sparse as spa
from jinja2 import Environment, PackageLoader, select_autoescape

_ALGEBRAS = (
    'cuda',
    'mkl',
    'builtin',
)  # Highest->Lowest priority of algebras that are tried in turn
# Mapping from algebra to loadable module
_ALGEBRA_MODULES = {
    'cuda': 'osqp_cuda',
    'mkl': 'osqp_mkl',
    'builtin': 'osqp.ext_builtin',
}
OSQP_ALGEBRA_BACKEND = os.environ.get('OSQP_ALGEBRA_BACKEND')  # If envvar is set, that algebra is used by default


def algebra_available(algebra):
    assert algebra in _ALGEBRAS, f'Unknown algebra {algebra}'
    module = _ALGEBRA_MODULES[algebra]

    try:
        importlib.import_module(module)
    except ImportError:
        return False
    else:
        return True


def algebras_available():
    return [algebra for algebra in _ALGEBRAS if algebra_available(algebra)]


def default_algebra():
    if OSQP_ALGEBRA_BACKEND is not None:
        return OSQP_ALGEBRA_BACKEND
    for algebra in _ALGEBRAS:
        if algebra_available(algebra):
            return algebra
    raise RuntimeError('No algebra backend available!')


def default_algebra_module():
    """
    Get the default algebra module.
    Note: importlib.import_module is cached so we pay almost no penalty
      for repeated calls to this function.
    """
    return importlib.import_module(_ALGEBRA_MODULES[default_algebra()])


def constant(which, algebra='builtin'):
    """
    Get a named constant from the extension module.
    Since constants are typically consistent across osqp algebras,
    we use the `builtin` algebra (always guaranteed to be available)
    by default.
    """
    m = importlib.import_module(_ALGEBRA_MODULES[algebra])
    _constant = getattr(m, which, None)

    if which in m.osqp_status_type.__members__:
        warnings.warn(
            'Direct access to osqp status values will be deprecated. Please use the SolverStatus enum instead.',
            PendingDeprecationWarning,
        )

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

        raise RuntimeError(f'Unknown constant {which}')


def construct_enum(name, binding_enum_name):
    """
    Dynamically construct an IntEnum from available enum members.
    For all values, see https://osqp.org/docs/interfaces/status_values.html
    """
    m = default_algebra_module()
    binding_enum = getattr(m, binding_enum_name)
    return IntEnum(name, [(v.name, v.value) for v in binding_enum.__members__.values()])


SolverStatus = construct_enum('SolverStatus', 'osqp_status_type')
SolverError = construct_enum('SolverError', 'osqp_error_type')


class OSQPException(Exception):
    """
    OSQPException is raised by the wrapper interface when it encounters an
    exception by the underlying OSQP solver.
    """

    def __init__(self, error_code=None):
        if error_code:
            self.args = (error_code,)

    def __eq__(self, error_code):
        return len(self.args) > 0 and self.args[0] == error_code


class OSQP:

    """
    For OSQP bindings (see bindings.cpp.in) that throw `ValueError`s
    (through `throw py::value_error(...)`), we catch and re-raise them
    as `OSQPException`s, with the correct int value as args[0].
    """

    @classmethod
    def raises_error(cls, fn, *args, **kwargs):
        try:
            return_value = fn(*args, **kwargs)
        except ValueError as e:
            if e.args:
                error_code = None
                try:
                    error_code = int(e.args[0])
                except ValueError:
                    pass
            raise OSQPException(error_code)
        else:
            return return_value

    def __init__(self, *args, **kwargs):
        self.m = None
        self.n = None

        self.algebra = kwargs.pop('algebra') if 'algebra' in kwargs else default_algebra()
        if not algebra_available(self.algebra):
            raise RuntimeError(f'Algebra {self.algebra} not available')
        self.ext = importlib.import_module(_ALGEBRA_MODULES[self.algebra])

        self._dtype = np.float32 if self.ext.OSQP_USE_FLOAT == 1 else np.float64
        self._itype = np.int64 if self.ext.OSQP_USE_LONG == 1 else np.int32

        # The following attributes are populated on setup()
        self._solver = None
        self._derivative_cache = {}

    def __str__(self):
        if self._solver is None:
            return f'Uninitialized OSQP with algebra={self.algebra}'
        else:
            return f'OSQP with algebra={self.algebra} ({self.solver_type})'

    def _infer_mnpqalu(self, P=None, q=None, A=None, l=None, u=None):
        # infer as many parameters of the problems as we can, and return them as a tuple
        if P is None:
            if q is not None:
                n = len(q)
            elif A is not None:
                n = A.shape[1]
            else:
                raise ValueError('The problem does not have any variables')
        else:
            n = P.shape[0]

        m = 0 if A is None else A.shape[0]

        if A is None:
            assert (l is None) and (u is None), 'If A is unspecified, leave l/u unspecified too.'
        else:
            assert (l is not None) or (u is not None), 'If A is specified, specify at least one of l/u.'
            if l is None:
                l = -np.inf * np.ones(A.shape[0])
            if u is None:
                u = np.inf * np.ones(A.shape[0])

        if P is None:
            P = spa.csc_matrix(
                (
                    np.zeros((0,), dtype=self._dtype),  # data
                    np.zeros((0,), dtype=self._itype),  # indices
                    np.zeros((n + 1,), dtype=self._itype),
                ),  # indptr
                shape=(n, n),
            )
        if q is None:
            q = np.zeros(n)

        if A is None:
            A = spa.csc_matrix(
                (
                    np.zeros((0,), dtype=self._dtype),  # data
                    np.zeros((0,), dtype=self._itype),  # indices
                    np.zeros((n + 1,), dtype=self._itype),
                ),  # indptr
                shape=(m, n),
            )
            l = np.zeros(A.shape[0])
            u = np.zeros(A.shape[0])

        assert len(q) == n, 'Incorrect dimension of q'
        assert len(l) == m, 'Incorrect dimension of l'
        assert len(u) == m, 'Incorrect dimension of u'

        if not spa.issparse(P) and isinstance(P, np.ndarray) and P.ndim == 2:
            raise TypeError('P is required to be a sparse matrix')
        if not spa.issparse(A) and isinstance(A, np.ndarray) and A.ndim == 2:
            raise TypeError('A is required to be a sparse matrix')

        if spa.tril(P, -1).data.size > 0:
            P = spa.triu(P, format='csc')

        # Convert matrices in CSC form to individual pointers
        if not spa.isspmatrix_csc(P):
            warnings.warn('Converting sparse P to a CSC matrix. This may take a while...')
            P = P.tocsc()
        if not spa.isspmatrix_csc(A):
            warnings.warn('Converting sparse A to a CSC matrix. This may take a while...')
            A = A.tocsc()

        if not P.has_sorted_indices:
            P.sort_indices()
        if not A.has_sorted_indices:
            A.sort_indices()

        u = np.minimum(u, self.constant('OSQP_INFTY'))
        l = np.maximum(l, -self.constant('OSQP_INFTY'))

        return m, n, P, q, A, l, u

    @property
    def capabilities(self):
        return int(self.ext.osqp_capabilities())

    def has_capability(self, capability: str):
        try:
            cap = int(self.ext.osqp_capabilities_type.__members__[capability])
        except KeyError:
            raise RuntimeError(f'Unrecognized capability {capability}')

        return (self.capabilities & cap) != 0

    @property
    def solver_type(self):
        return (
            'direct'
            if self.settings.linsys_solver == self.ext.osqp_linsys_solver_type.OSQP_DIRECT_SOLVER
            else 'indirect'
        )

    @property
    def cg_preconditioner(self):
        return 'diagonal' if self.settings.cg_precond == self.ext.OSQP_DIAGONAL_PRECONDITIONER else None

    def _as_dense(self, m):
        assert isinstance(m, self.ext.CSC)
        _m_csc = spa.csc_matrix((m.x, m.i, m.p))
        return np.array(_m_csc.todense())

    def _csc_triu_as_csc_full(self, m):
        _m_triu_dense = self._as_dense(m)
        _m_full_dense = np.tril(_m_triu_dense.T, -1) + _m_triu_dense
        _m_full_csc = spa.csc_matrix(_m_full_dense)
        return self.ext.CSC(_m_full_csc)

    def constant(self, which):
        return constant(which, algebra=self.algebra)

    def update_settings(self, **kwargs):
        assert self.settings is not None

        # Some setting names have changed. Support the old names for now, but warn the caller.
        renamed_settings = {
            'polish': 'polishing',
            'warm_start': 'warm_starting',
        }
        for k, v in renamed_settings.items():
            if k in kwargs:
                warnings.warn(
                    f'"{k}" is deprecated. Please use "{v}" instead.',
                    DeprecationWarning,
                )
                kwargs[v] = kwargs[k]
                del kwargs[k]

        settings_changed = False

        if 'rho' in kwargs and self._solver is not None:
            self._solver.update_rho(kwargs.pop('rho'))
        if 'solver_type' in kwargs:
            value = kwargs.pop('solver_type')
            assert value in ('direct', 'indirect')
            self.settings.linsys_solver = (
                self.ext.osqp_linsys_solver_type.OSQP_DIRECT_SOLVER
                if value == 'direct'
                else self.ext.osqp_linsys_solver_type.OSQP_INDIRECT_SOLVER
            )
            settings_changed = True
        if 'cg_preconditioner' in kwargs:
            value = kwargs.pop('cg_preconditioner')
            assert value in (None, 'diagonal')
            self.settings.cg_precond = (
                self.ext.OSQP_DIAGONAL_PRECONDITIONER if value == 'diagonal' else self.ext.OSQP_NO_PRECONDITIONER
            )
            settings_changed = True

        for k in self.ext.OSQPSettings.__dict__:
            if not k.startswith('__'):
                if k in kwargs:
                    setattr(self.settings, k, kwargs.pop(k))
                    settings_changed = True

        if kwargs:
            raise ValueError(f'Unrecognized settings {list(kwargs.keys())}')

        if settings_changed and self._solver is not None:
            self.raises_error(self._solver.update_settings, self.settings)

    def update(self, **kwargs):
        # TODO: sanity-check on types/dimensions

        q, l, u = kwargs.get('q'), kwargs.get('l'), kwargs.get('u')
        if l is not None:
            l = np.maximum(l, -self.constant('OSQP_INFTY'))
        if u is not None:
            u = np.minimum(u, self.constant('OSQP_INFTY'))

        if q is not None or l is not None or u is not None:
            self._solver.update_data_vec(q=q, l=l, u=u)
        if 'Px' in kwargs or 'Px_idx' in kwargs or 'Ax' in kwargs or 'Ax_idx' in kwargs:
            self._solver.update_data_mat(
                P_x=kwargs.get('Px'),
                P_i=kwargs.get('Px_idx'),
                A_x=kwargs.get('Ax'),
                A_i=kwargs.get('Ax_idx'),
            )

        if q is not None:
            self._derivative_cache['q'] = q
        if l is not None:
            self._derivative_cache['l'] = l
        if u is not None:
            self._derivative_cache['u'] = u

        for _var in ('P', 'A'):
            _varx = f'{_var}x'
            if kwargs.get(_varx) is not None:
                if kwargs.get(f'{_varx}_idx') is None:
                    self._derivative_cache[_var].data = kwargs[_varx]
                else:
                    self._derivative_cache[_var].data[kwargs[f'{_varx}_idx']] = kwargs[_varx]

        # delete results from self._derivative_cache to prohibit
        # taking the derivative of unsolved problems
        self._derivative_cache.pop('results', None)
        self._derivative_cache.pop('solver', None)
        self._derivative_cache.pop('M', None)

    def setup(self, P, q, A, l, u, **settings):
        m, n, P, q, A, l, u = self._infer_mnpqalu(P=P, q=q, A=A, l=l, u=u)
        self._derivative_cache.update({'P': P, 'q': q, 'A': A, 'l': l, 'u': u})
        self.m = m
        self.n = n
        P = self.ext.CSC(P.astype(self._dtype))
        q = q.astype(self._dtype)
        A = self.ext.CSC(A.astype(self._dtype))
        l = l.astype(self._dtype)
        u = u.astype(self._dtype)

        self.settings = self.ext.OSQPSettings()
        self.ext.osqp_set_default_settings(self.settings)
        self.update_settings(**settings)

        self._solver = self.raises_error(
            self.ext.OSQPSolver,
            P,
            q,
            A,
            l,
            u,
            self.m,
            self.n,
            self.settings,
        )
        if 'rho' in settings:
            self._solver.update_rho(settings['rho'])

    def warm_start(self, x=None, y=None):
        # TODO: sanity checks on types/dimensions
        return self._solver.warm_start(x, y)

    def solve(self, raise_error=None):
        if raise_error is None:
            warnings.warn(
                'The default value of raise_error will change to True in the future.',
                PendingDeprecationWarning,
            )
            raise_error = False

        self._solver.solve()

        info = self._solver.info
        if info.status_val == SolverStatus.OSQP_NON_CVX:
            info.obj_val = np.nan

        if info.status_val != SolverStatus.OSQP_SOLVED and raise_error:
            raise OSQPException(info.status_val)

        # Create a Namespace of OSQPInfo keys and associated values
        _info = SimpleNamespace(**{k: getattr(info, k) for k in info.__class__.__dict__ if not k.startswith('__')})

        # TODO: The following structure is only to maintain backward compatibility, where x/y are attributes
        # directly inside the returned object on solve(). This should be simplified!
        results = SimpleNamespace(
            x=self._solver.solution.x,
            y=self._solver.solution.y,
            prim_inf_cert=self._solver.solution.prim_inf_cert,
            dual_inf_cert=self._solver.solution.dual_inf_cert,
            info=_info,
        )

        self._derivative_cache['results'] = results
        return results

    def _render_pywrapper_files(self, output_folder, **kwargs):
        env = Environment(
            loader=PackageLoader('osqp.codegen.pywrapper', package_path=''),
            autoescape=select_autoescape(),
        )

        for template_name in env.list_templates(extensions='.jinja'):
            template = env.get_template(template_name)
            template_base_name = os.path.splitext(template_name)[0]

            with open(os.path.join(output_folder, template_base_name), 'w') as f:
                f.write(template.render(**kwargs))

    def codegen(
        self,
        folder,
        parameters='vectors',
        extension_name='emosqp',
        force_rewrite=False,
        use_float=False,
        printing_enable=False,
        profiling_enable=False,
        interrupt_enable=False,
        include_codegen_src=True,
        prefix='',
        compile=False,
    ):
        assert self.has_capability('OSQP_CAPABILITY_CODEGEN'), 'This OSQP object does not support codegen'
        assert parameters in (
            'vectors',
            'matrices',
        ), 'Unknown parameters specification'

        defines = self.ext.OSQPCodegenDefines()
        self.ext.osqp_set_default_codegen_defines(defines)

        defines.embedded_mode = 1 if parameters == 'vectors' else 2
        defines.float_type = 1 if use_float else 0
        defines.printing_enable = 1 if printing_enable else 0
        defines.profiling_enable = 1 if profiling_enable else 0
        defines.interrupt_enable = 1 if interrupt_enable else 0
        defines.derivatives_enable = 0

        folder = os.path.abspath(folder)
        if include_codegen_src:
            # https://github.com/python/importlib_resources/issues/85
            try:
                codegen_src_path = importlib.resources.files('osqp.codegen').joinpath('codegen_src')
                shutil.copytree(codegen_src_path, folder, dirs_exist_ok=force_rewrite)
            except AttributeError:
                handle = importlib.resources.path('osqp.codegen', 'codegen_src')
                with handle as codegen_src_path:
                    shutil.copytree(codegen_src_path, folder, dirs_exist_ok=force_rewrite)

        # The C codegen call expects the folder to exist and have a trailing slash
        os.makedirs(folder, exist_ok=True)
        if not folder.endswith(os.path.sep):
            folder += os.path.sep

        status = self._solver.codegen(folder, prefix, defines)
        assert status == 0, f'Codegen failed with error code {status}'

        if extension_name is not None:
            assert include_codegen_src, 'If generating python wrappers, include_codegen_src must be True'
            template_vars = dict(
                prefix=prefix,
                extension_name=extension_name,
                embedded_mode=defines.embedded_mode,
            )
            self._render_pywrapper_files(folder, **template_vars)
            if compile:
                subprocess.check_call(
                    [
                        sys.executable,
                        'setup.py',
                        'build_ext',
                        '--inplace',
                    ],
                    cwd=folder,
                )

        return folder

    def adjoint_derivative_compute(self, dx=None, dy=None):
        """
        Compute adjoint derivative after solve.
        """

        assert self.has_capability('OSQP_CAPABILITY_DERIVATIVES'), 'This OSQP object does not support derivatives'

        try:
            results = self._derivative_cache['results']
        except KeyError:
            raise ValueError(
                'Problem has not been solved. ' 'You cannot take derivatives. ' 'Please call the solve function.'
            )

        if results.info.status_val != SolverStatus.OSQP_SOLVED:
            raise ValueError('Problem has not been solved to optimality. ' 'You cannot take derivatives')

        if dy is None:
            dy = np.zeros(self.m)

        self._solver.adjoint_derivative_compute(dx, dy)

    def adjoint_derivative_get_mat(self, as_dense=True, dP_as_triu=True):
        """
        Get dP/dA matrices after an invocation of adjoint_derivative_compute
        """

        assert self.has_capability('OSQP_CAPABILITY_DERIVATIVES'), 'This OSQP object does not support derivatives'

        try:
            results = self._derivative_cache['results']
        except KeyError:
            raise ValueError(
                'Problem has not been solved. ' 'You cannot take derivatives. ' 'Please call the solve function.'
            )

        if results.info.status_val != SolverStatus.OSQP_SOLVED:
            raise ValueError('Problem has not been solved to optimality. ' 'You cannot take derivatives')

        P, _ = self._derivative_cache['P'], self._derivative_cache['q']
        A = self._derivative_cache['A']

        dP = self.ext.CSC(P.copy())
        dA = self.ext.CSC(A.copy())

        self._solver.adjoint_derivative_get_mat(dP, dA)

        if not dP_as_triu:
            dP = self._csc_triu_as_csc_full(dP)

        if as_dense:
            dP = self._as_dense(dP)
            dA = self._as_dense(dA)

        return dP, dA

    def adjoint_derivative_get_vec(self):
        """
        Get dq/dl/du vectors after an invocation of adjoint_derivative_compute
        """

        assert self.has_capability('OSQP_CAPABILITY_DERIVATIVES'), 'This OSQP object does not support derivatives'

        try:
            results = self._derivative_cache['results']
        except KeyError:
            raise ValueError(
                'Problem has not been solved. ' 'You cannot take derivatives. ' 'Please call the solve function.'
            )

        if results.info.status_val != SolverStatus.OSQP_SOLVED:
            raise ValueError('Problem has not been solved to optimality. ' 'You cannot take derivatives')

        dq = np.empty(self.n).astype(self._dtype)
        dl = np.zeros(self.m).astype(self._dtype)
        du = np.zeros(self.m).astype(self._dtype)

        self._solver.adjoint_derivative_get_vec(dq, dl, du)

        return dq, dl, du
