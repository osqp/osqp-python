import os
import importlib
import shutil
from types import SimpleNamespace
import warnings
import numpy as np
import scipy.sparse as spa
from osqp import algebra_available, default_algebra
from osqp.interface import constant, _ALGEBRA_MODULES


class OSQP:

    @staticmethod
    def _infer_mnpqalu(P=None, q=None, A=None, l=None, u=None):
        # infer as many parameters of the problems as we can, and return them as a tuple
        if P is None:
            if q is not None:
                n = len(q)
            elif A is not None:
                n = A.shape[1]
            else:
                raise ValueError("The problem does not have any variables")
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
                (np.zeros((0,), dtype=np.double),    # data
                 np.zeros((0,), dtype=np.int),       # indices
                 np.zeros((n + 1,), dtype=np.int)),  # indptr
                shape=(n, n)
            )
        if q is None:
            q = np.zeros(n)

        if A is None:
            A = spa.csc_matrix(
                (np.zeros((0,), dtype=np.double),    # data
                 np.zeros((0,), dtype=np.int),       # indices
                 np.zeros((n + 1,), dtype=np.int)),  # indptr
                shape=(m, n)
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

        # Convert matrices in CSC form and to individual pointers
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

        u = np.minimum(u, constant('OSQP_INFTY'))
        l = np.maximum(l, -constant('OSQP_INFTY'))

        return m, n, P, q, A, l, u

    def __init__(self, *args, **kwargs):
        self.m = None
        self.n = None
        self.P = None
        self.q = None
        self.A = None
        self.l = None
        self.u = None

        self.algebra = kwargs.pop('algebra', default_algebra())
        if not algebra_available(self.algebra):
            raise RuntimeError(f'Algebra {self.algebra} not available')
        self.ext = importlib.import_module(_ALGEBRA_MODULES[self.algebra])

        self.settings = self.ext.OSQPSettings()
        self.ext.osqp_set_default_settings(self.settings)

        self._dtype = np.float32 if self.ext.OSQP_DFLOAT == 1 else np.float64
        self._itype = np.int64 if self.ext.OSQP_DLONG == 1 else np.int32

        # The following attributes are populated on setup()
        self._solver = None
        self._derivative_cache = {}

    def __str__(self):
        return f'OSQP with algebra={self.algebra}'

    @property
    def solver_type(self):
        return 'direct' if self.settings.linsys_solver == self.ext.osqp_linsys_solver_type.OSQP_DIRECT_SOLVER else 'indirect'

    @solver_type.setter
    def solver_type(self, value):
        assert value in ('direct', 'indirect')
        self.settings.linsys_solver = self.ext.osqp_linsys_solver_type.OSQP_DIRECT_SOLVER if value == 'direct' else self.ext.osqp_linsys_solver_type.OSQP_INDIRECT_SOLVER

    def _as_dense(self, m):
        assert isinstance(m, self.ext.CSC)
        _m_csc = spa.csc_matrix((m.x, m.i, m.p))
        return np.array(_m_csc.todense())

    def constant(self, which):
        return constant(which, algebra=self.algebra)

    def update_settings(self, **kwargs):

        # Some setting names have changed. Support the old names for now, but warn the caller.
        renamed_settings = {'polish': 'polishing', 'warm_start': 'warm_starting'}
        for k, v in renamed_settings.items():
            if k in kwargs:
                warnings.warn(f'"{k}" is deprecated. Please use "{v}" instead.', DeprecationWarning)
                kwargs[v] = kwargs[k]
                del kwargs[k]

        new_settings = self.ext.OSQPSettings()
        for k in self.ext.OSQPSettings.__dict__:
            if not k.startswith('__'):
                if k in kwargs:
                    setattr(new_settings, k, kwargs[k])
                else:
                    setattr(new_settings, k, getattr(self.settings, k))

        if self._solver is not None:
            if 'rho' in kwargs:
                self._solver.update_rho(kwargs.pop('rho'))
            if kwargs:
                self._solver.update_settings(new_settings)
            self.settings = self._solver.get_settings()  # TODO: Why isn't this just an attribute?
        else:
            self.settings = new_settings

    def update(self, **kwargs):
        # TODO: sanity-check on types/dimensions

        q, l, u = kwargs.get('q'), kwargs.get('l'), kwargs.get('u')
        if l is not None:
            l = np.maximum(l, -constant('OSQP_INFTY'))
        if u is not None:
            u = np.minimum(u, constant('OSQP_INFTY'))

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
        self.m = m
        self.n = n
        self.P = self.ext.CSC(P.astype(self._dtype))
        self.q = q.astype(self._dtype)
        self.A = self.ext.CSC(A.astype(self._dtype))
        self.l = l.astype(self._dtype)
        self.u = u.astype(self._dtype)

        self.update_settings(**settings)

        self._solver = self.ext.OSQPSolver(self.P, self.q, self.A, self.l, self.u, self.m, self.n, self.settings)
        self._derivative_cache.update({
            'P': P,
            'q': q,
            'A': A,
            'l': l,
            'u': u
        })

    def warm_start(self, x=None, y=None):
        # TODO: sanity checks on types/dimensions
        return self._solver.warm_start(x, y)

    def solve(self, raise_error=False):
        self._solver.solve()

        info = self._solver.info
        if info.status_val == constant('OSQP_NON_CVX', algebra=self.algebra):
            info.obj_val = np.nan
        # TODO: Handle primal/dual infeasibility

        if info.status_val != constant('OSQP_SOLVED') and raise_error:
            raise ValueError('Problem not solved!')

        # Create a Namespace of OSQPInfo keys and associated values
        _info = SimpleNamespace(**{k: getattr(info, k) for k in info.__class__.__dict__ if not k.startswith('__')})

        # TODO: The following structure is only to maintain backward compatibility, where x/y are attributes
        # directly inside the returned object on solve(). This should be simplified!
        results = SimpleNamespace(
            x=self._solver.solution.x,
            y=self._solver.solution.y,
            info=_info
        )

        self._derivative_cache['results'] = results
        return results

    def codegen(self, folder, project_type='', parameters='vectors', python_ext_name='emosqp', force_rewrite=False,
                FLOAT=False, LONG=False, prefix='', compile=False):

        assert project_type in (None, ''), 'project_type should be blank/None, and is only provided for backwards API compatibility'
        assert parameters in ('vectors', 'matrices'), 'Unknown parameters specification'
        assert not LONG, 'Long ("long long" in C) is no longer supported in codegen. We only support C89 compliant version of the long ints'

        defines = self.ext.OSQPCodegenDefines()
        defines.embedded_mode = 1 if parameters == 'vectors' else 2
        defines.float_type = 1 if FLOAT else 0
        defines.printing_enable = 0
        defines.profiling_enable = 0
        defines.interrupt_enable = 0

        folder = os.path.abspath(folder)
        with importlib.resources.path('osqp.new_codegen', 'codegen_src') as codegen_src_path:
            shutil.copytree(codegen_src_path, folder, dirs_exist_ok=force_rewrite)

        # The C codegen call expects the folder to exist and have a trailing slash
        if not folder.endswith(os.path.sep):
            folder += os.path.sep

        status = self._solver.codegen(folder, prefix, defines)
        if status != 0:
            raise RuntimeError(f'Codegen failed with error code {status}')

        if compile:
            raise NotImplementedError

    def adjoint_derivative(self, dx=None, dy_u=None, dy_l=None, as_dense=True):
        """
        Compute adjoint derivative after solve.
        """

        try:
            results = self._derivative_cache['results']
        except KeyError:
            raise ValueError("Problem has not been solved. "
                             "You cannot take derivatives. "
                             "Please call the solve function.")

        if results.info.status != "solved":
            raise ValueError("Problem has not been solved to optimality. "
                             "You cannot take derivatives")

        P, q = self._derivative_cache['P'], self._derivative_cache['q']
        A = self._derivative_cache['A']
        m, n = A.shape

        if dy_u is None:
            dy_u = np.zeros(m)
        if dy_l is None:
            dy_l = np.zeros(m)

        dP = self.ext.CSC(P.copy())
        dq = np.empty(n).astype(self._dtype)
        dA = self.ext.CSC(A.copy())
        dl = np.zeros(m).astype(self._dtype)
        du = np.zeros(m).astype(self._dtype)

        # In the following call to the C extension, the first 3 are inputs, the remaining are outputs
        self._solver.adjoint_derivative(dx, dy_l, dy_u, dP, dq, dA, dl, du)

        if as_dense:
            dP = self._as_dense(dP)
            dA = self._as_dense(dA)

        return dP, dq, dA, dl, du
