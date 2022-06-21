import sys
import os
import importlib
from types import SimpleNamespace
import warnings
import numpy as np
import scipy.sparse as spa
import qdldl
from osqp import algebra_available, default_algebra
from osqp.interface import constant, _ALGEBRA_MODULES
import osqp.utils as utils
import osqp.codegen as cg


class OSQP:
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
        self.m = l.shape[0]
        self.n = q.shape[0]
        self.P = self.ext.CSC(spa.triu(P.astype(self._dtype), format='csc'))
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

        # The C codegen call expects the folder to exist and have a trailing slash
        folder = os.path.abspath(folder)
        os.makedirs(folder, exist_ok=force_rewrite)
        if not folder.endswith(os.path.sep):
            folder += os.path.sep

        status = self._solver.codegen(folder, prefix, defines)
        if status != 0:
            raise RuntimeError(f'Codegen failed with error code {status}')

        if compile:
            raise NotImplementedError

    def derivative_iterative_refinement(self, rhs, max_iter, tol):
        M = self._derivative_cache['M']

        # Prefactor
        solver = self._derivative_cache['solver']

        sol = solver.solve(rhs)

        for k in range(max_iter):
            if np.linalg.norm(M @ sol - rhs) < tol:
                break
            delta_sol = solver.solve(M @ sol - rhs)
            sol = sol - delta_sol

        if k == max_iter - 1:
            warnings.warn("max_iter iterative refinement reached.")

        return sol

    def adjoint_derivative(self, dx=None, dy_u=None, dy_l=None,
                           P_idx=None, A_idx=None, **kwargs):
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

        x = results.x
        y = results.y
        y_u = np.maximum(y, 0)
        y_l = -np.minimum(y, 0)

        out_dict = self.derivative_setup(x, y)
        M, P, q, A = out_dict['M'], out_dict['P'], out_dict['q'], out_dict['A']
        l, u, G, h = out_dict['l'], out_dict['u'], out_dict['G'], out_dict['h']
        l_non_inf, u_non_inf = out_dict['l_non_inf'], out_dict['u_non_inf']
        num_eq, num_ineq = out_dict['num_eq'], out_dict['num_ineq']
        lambd, nu = out_dict['lambd'], out_dict['nu']
        eq_indices, ineq_indices = out_dict['eq_indices'], out_dict['ineq_indices']
        y_ineq, y_l_ineq, y_u_ineq = out_dict['y_ineq'], out_dict['y_l_ineq'], out_dict['y_u_ineq']

        m, n = A.shape

        if A_idx is None:
            A_idx = A.nonzero()
        if P_idx is None:
            P_idx = P.nonzero()
        if dy_u is None:
            dy_u = np.zeros(m)
        if dy_l is None:
            dy_l = np.zeros(m)

        dy_l_ineq = dy_l[ineq_indices].copy()
        dy_u_ineq = dy_u[ineq_indices].copy()

        dy_l_eq = dy_l[eq_indices]
        dy_u_eq = dy_u[eq_indices]
        dnu = np.zeros(eq_indices.size)

        if eq_indices.size > 0:
            dnu[nu >= 0] = dy_u_eq[nu >= 0]
            dnu[nu < 0] = -dy_l_eq[nu < 0]
        dlambd = np.concatenate([dy_l_ineq[l_non_inf], dy_u_ineq[u_non_inf]])

        rhs = - np.concatenate([dx, dlambd, dnu])
        
        if 'eps_iter_ref' in kwargs:
            eps_iter_ref = kwargs['eps_iter_ref']
        else:
            eps_iter_ref = 1e-6
        if 'max_iter' in kwargs:
            max_iter = kwargs['max_iter']
        else:
            max_iter = 200
        if 'tol' in kwargs:
            tol = kwargs['tol']
        else:
            tol = 1e-12
        B = spa.bmat([
            [spa.eye(n + num_ineq + num_eq), M.T],
            [M, None]
        ])
        delta_B = spa.bmat([[eps_iter_ref * spa.eye(n + num_ineq + num_eq), None],
                            [None, -eps_iter_ref * spa.eye(n + num_ineq + num_eq)]],
                            format='csc')
        if self._derivative_cache.get('solver') is None:
            solver = qdldl.Solver(B + delta_B)
            self._derivative_cache['M'] = B
            self._derivative_cache['solver'] = solver
        rhs_b = np.concatenate([rhs, np.zeros(n + num_ineq + num_eq)])
        r_sol_b = self.derivative_iterative_refinement(
            rhs_b, max_iter, tol)
        dual, primal = np.split(r_sol_b, [n + num_ineq + num_eq])
        

        r_x_b, r_lambda_l_b, r_lambda_u_b, r_nu = np.split(
            primal, [n, n + l_non_inf.size, n + num_ineq])
        r_x, r_lambda_l, r_lambda_u = r_x_b, r_lambda_l_b, r_lambda_u_b

        # revert back to y form
        r_yu = np.zeros(m)
        r_yl = np.zeros(m)
        r_yu_ineq = np.zeros(m - num_eq)
        r_yl_ineq = np.zeros(m - num_eq)
        r_yu_ineq[u_non_inf] = r_lambda_u
        r_yl_ineq[l_non_inf] = -r_lambda_l

        r_yu[ineq_indices] = r_yu_ineq
        r_yl[ineq_indices] = r_yl_ineq

        # go from (r_nu, r_yu, r_yl) to (r_yu, r_yl)
        r_yu_eq = r_nu.copy() / nu
        r_yu_eq[nu < 0] = 0
        r_yu[eq_indices] = r_yu_eq

        r_yl_eq = -r_nu / (nu)
        r_yl_eq[nu >= 0] = 0
        r_yl[eq_indices] = r_yl_eq

        # Extract derivatives for the constraints
        rows, cols = A_idx
        ryu = spa.diags(y_u) @ r_yu
        ryl = -spa.diags(y_l) @ r_yl
        dA_vals = (y_u[rows] - y_l[rows]) * r_x[cols] + \
                  (ryu[rows] - ryl[rows]) * x[cols]
        dA = spa.csc_matrix((dA_vals, (rows, cols)), shape=A.shape)

        du = -ryu
        dl = ryl

        # Extract derivatives for the cost (P, q)
        rows, cols = P_idx
        dP_vals = .5 * (r_x[rows] * x[cols] + r_x[cols] * x[rows])
        dP = spa.csc_matrix((dP_vals, P_idx), shape=P.shape)
        dq = r_x

        # -------- CHECK ---------
        _dP = self.ext.CSC(P.copy())
        _dq = np.empty(n).astype(self._dtype)
        _dA = self.ext.CSC(A.copy())
        _dl = np.zeros(len(r_yl)).astype(self._dtype)
        _du = np.zeros(len(r_yu)).astype(self._dtype)

        # In the following call to the C extension, the first 3 are inputs, the remaining are outputs
        self._solver.adjoint_derivative(dx, dy_l, dy_u, _dP, _dq, _dA, _dl, _du)

        tol = 0.0001
        assert np.allclose(_dl, dl, atol=tol)
        assert np.allclose(_du, du, atol=tol)
        assert np.allclose(_dq, dq, atol=tol)
        assert np.all(_dP.i == dP.indices)
        assert np.all(_dP.p == dP.indptr)
        assert np.allclose(_dP.x, dP.data, atol=tol)
        assert np.all(_dA.i == dA.indices)
        assert np.all(_dA.p == dA.indptr)
        assert np.allclose(_dA.x, dA.data, atol=tol)
        # -------- CHECK ---------

        return dP, dq, dA, dl, du

    def forward_derivative(self, dP=None, dq=None, dA=None, dl=None, du=None,
                           P_idx=None, A_idx=None, **kwargs):
        """
        Compute forward derivative after solve.
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
        l, u = self._derivative_cache['l'], self._derivative_cache['u']

        x = results.x
        y = results.y

        if A_idx is None:
            A_idx = A.nonzero()
        if P_idx is None:
            P_idx = P.nonzero()
        m, n = A.shape

        if dP is None:
            dP = np.zeros((n, n))
        if dq is None:
            dq = np.zeros(n)
        if dA is None:
            dA = np.zeros((m, n))
        if dl is None:
            dl = np.zeros(m)
        if du is None:
            du = np.zeros(m)

        # M, P, q, A, l, u, A_eq, b, G, h, l_non_inf, u_non_inf, num_eq, num_ineq, lambd, nu, eq_indices, ineq_indices, y_ineq, y_l_ineq, y_u_ineq = self.derivative_setup(
        #     x, y)
        out_dict = self.derivative_setup(x, y)
        M, P, q, A = out_dict['M'], out_dict['P'], out_dict['q'], out_dict['A']
        l, u, G, h = out_dict['l'], out_dict['u'], out_dict['G'], out_dict['h']
        l_non_inf, u_non_inf = out_dict['l_non_inf'], out_dict['u_non_inf']
        num_eq, num_ineq = out_dict['num_eq'], out_dict['num_ineq']
        lambd, nu = out_dict['lambd'], out_dict['nu']
        eq_indices, ineq_indices = out_dict['eq_indices'], out_dict['ineq_indices']
        y_ineq, y_l_ineq, y_u_ineq = out_dict['y_ineq'], out_dict['y_l_ineq'], out_dict['y_u_ineq']

        dA_ineq = spa.csc_matrix(dA[ineq_indices, :])
        dl_ineq = dl[ineq_indices]
        du_ineq = du[ineq_indices]
        dA_eq = dA[eq_indices, :]
        db = du[eq_indices]

        dG = spa.bmat([
            [-dA_ineq[l_non_inf, :]],
            [dA_ineq[u_non_inf, :]],
        ])
        dh = np.concatenate([-dl_ineq[l_non_inf], du_ineq[u_non_inf]])

        # form g
        dia_lambda = spa.diags(lambd)
        g1 = dP @ x + dq + dG.T @ lambd + dA_eq.T @ nu
        g2 = dia_lambda @ (dG @ x - dh)
        g3 = dA_eq @ x - db
        g = np.concatenate([g1, g2, g3])
        rhs = -g

        B = spa.bmat([
            [spa.eye(n + num_ineq + num_eq), M],
            [M.T, None]
        ])
        if 'eps_iter_ref' in kwargs:
            eps_iter_ref = kwargs['eps_iter_ref']
        else:
            eps_iter_ref = 1e-6
        if 'max_iter' in kwargs:
            max_iter = kwargs['max_iter']
        else:
            max_iter = 20
        if 'tol' in kwargs:
            tol = kwargs['tol']
        else:
            tol = 1e-12
        delta_B = spa.bmat([[eps_iter_ref * spa.eye(n + num_ineq + num_eq), None],
                            [None, -eps_iter_ref * spa.eye(n + num_ineq + num_eq)]],
                            format='csc')
        
        solver = qdldl.Solver(B + delta_B)
        self._derivative_cache['M'] = B
        self._derivative_cache['solver'] = solver
        rhs_b = np.concatenate([rhs, np.zeros(n + num_ineq + num_eq)])

        r_sol_b = self.derivative_iterative_refinement(
            rhs_b, max_iter, tol)
        dual, primal = np.split(r_sol_b, [n + num_ineq + num_eq])

        dx, dlambda_l, dlambda_u, dnu = np.split(
            primal, [n, n + l_non_inf.size, n + num_ineq])

        dyl = np.zeros(m)
        dyu = np.zeros(m)

        # get eq part of dyl, dyu
        dyl_eq = np.zeros(eq_indices.size)
        dyu_eq = np.zeros(eq_indices.size)
        if eq_indices.size > 0:
            y_eq = y[eq_indices]
            dyl_eq[y_eq < 0] = -dnu[y_eq < 0]
            dyu_eq[y_eq >= 0] = dnu[y_eq >= 0]

        # get ineq part of dyl, dyu
        dyl_ineq = dlambda_l
        dyu_ineq = dlambda_u

        dyl[eq_indices] = dyl_eq
        dyl[ineq_indices[l_non_inf]] = dyl_ineq
        dyu[eq_indices] = dyu_eq
        dyu[ineq_indices[u_non_inf]] = dyu_ineq

        return dx, dyl, dyu

    def derivative_setup(self, x, y):
        P, q = self._derivative_cache['P'], self._derivative_cache['q']
        A = self._derivative_cache['A']
        l, u = self._derivative_cache['l'], self._derivative_cache['u']

        # identify equality constraints
        eq_indices = np.where(l == u)[0]
        ineq_indices = np.where(l < u)[0]
        num_eq = eq_indices.size
        A_ineq = A[ineq_indices, :]
        l_ineq = l[ineq_indices]
        u_ineq = u[ineq_indices]
        A_eq = A[eq_indices, :]
        b = u[eq_indices]

        # switch to Gx <= h form
        l_non_inf = np.where(l_ineq > -constant('OSQP_INFTY'))[0]
        u_non_inf = np.where(u_ineq < constant('OSQP_INFTY'))[0]

        num_ineq = l_non_inf.size + u_non_inf.size
        G = spa.bmat([
            [-A_ineq[l_non_inf, :]],
            [A_ineq[u_non_inf, :]],
        ])
        h = np.concatenate([-l_ineq[l_non_inf], u_ineq[u_non_inf]])

        y_ineq = y[ineq_indices].copy()
        y_u_ineq = np.maximum(y_ineq, 0)
        y_l_ineq = -np.minimum(y_ineq, 0)
        lambd = np.concatenate([y_l_ineq[l_non_inf], y_u_ineq[u_non_inf]])
        nu = y[eq_indices]

        dia_lambda = spa.diags(lambd)
        slacks = G @ x - h

        M = spa.bmat([
            [P, G.T, A_eq.T],
            [dia_lambda @ G, spa.diags(slacks), None],
            [A_eq, None, None]
        ], format='csc')
        out_dict = {'M': M, 'P': P, 'q': q, 'A': A, 'l': l, 'u': u, 
                    'A_eq': A_eq, 'b': b, 'G': G, 'h': h, 'l_non_inf': l_non_inf,
                    'u_non_inf': u_non_inf, 'num_eq': num_eq, 'num_ineq': num_ineq,
                    'lambd': lambd, 'nu': nu, 'eq_indices': eq_indices, 'ineq_indices': ineq_indices, 
                    'y_ineq': y_ineq, 'y_l_ineq': y_l_ineq, 'y_u_ineq': y_u_ineq}
        return out_dict
        # return M, P, q, A, l, u, A_eq, b, G, h, l_non_inf, u_non_inf, num_eq, num_ineq, lambd, nu, eq_indices, ineq_indices, y_ineq, y_l_ineq, y_u_ineq
