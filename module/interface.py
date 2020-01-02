"""
Python interface module for OSQP solver v0.6.1
"""
from __future__ import print_function
from builtins import object
import osqp._osqp as _osqp  # Internal low level module
import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg as sla
from platform import system
import osqp.codegen as cg
import osqp.utils as utils
from sksparse.cholmod import cholesky
import sys


class OSQP(object):
    def __init__(self):
        self._model = _osqp.OSQP()

    def version(self):
        return self._model.version()

    def setup(self, P=None, q=None, A=None, l=None, u=None, **settings):
        """
        Setup OSQP solver problem of the form

        minimize     1/2 x' * P * x + q' * x
        subject to   l <= A * x <= u

        solver settings can be specified as additional keyword arguments
        """
        # TODO(bart): this will be unnecessary when the derivative will be in C
        self._derivative_cache = {'P': P, 'q': q, 'A': A, 'l': l, 'u': u}

        unpacked_data, settings = utils.prepare_data(P, q, A, l, u, **settings)
        self._model.setup(*unpacked_data, **settings)

    def update(self, q=None, l=None, u=None,
               Px=None, Px_idx=np.array([]), Ax=None, Ax_idx=np.array([])):
        """
        Update OSQP problem arguments
        """

        # get problem dimensions
        (n, m) = self._model.dimensions()

        # check consistency of the input arguments
        if q is not None and len(q) != n:
            raise ValueError("q must have length n")
        if l is not None:
            if not isinstance(l, np.ndarray):
                raise TypeError("l must be numpy.ndarray, not %s" %
                                type(l).__name__)
            elif len(l) != m:
                raise ValueError("l must have length m")
            # Convert values to -OSQP_INFTY
            l = np.maximum(l, -_osqp.constant('OSQP_INFTY'))
        if u is not None:
            if not isinstance(u, np.ndarray):
                raise TypeError("u must be numpy.ndarray, not %s" %
                                type(u).__name__)
            elif len(u) != m:
                raise ValueError("u must have length m")
            # Convert values to OSQP_INFTY
            u = np.minimum(u, _osqp.constant('OSQP_INFTY'))
        if Ax is None:
            if len(Ax_idx) > 0:
                raise ValueError("Vector Ax has not been specified")
        else:
            if len(Ax_idx) > 0 and len(Ax) != len(Ax_idx):
                raise ValueError("Ax and Ax_idx must have the same lengths")
        if Px is None:
            if len(Px_idx) > 0:
                raise ValueError("Vector Px has not been specified")
        else:
            if len(Px_idx) > 0 and len(Px) != len(Px_idx):
                raise ValueError("Px and Px_idx must have the same lengths")
        if q is None and l is None and u is None and Px is None and Ax is None:
            raise ValueError("No updatable data has been specified")

        # update linear cost
        if q is not None:
            self._model.update_lin_cost(q)

        # update lower bound
        if l is not None and u is None:
            self._model.update_lower_bound(l)

        # update upper bound
        if u is not None and l is None:
            self._model.update_upper_bound(u)

        # update bounds
        if l is not None and u is not None:
            self._model.update_bounds(l, u)

        # update matrix P
        if Px is not None and Ax is None:
            self._model.update_P(Px, Px_idx, len(Px))

        # update matrix A
        if Ax is not None and Px is None:
            self._model.update_A(Ax, Ax_idx, len(Ax))

        # update matrices P and A
        if Px is not None and Ax is not None:
            self._model.update_P_A(Px, Px_idx, len(Px), Ax, Ax_idx, len(Ax))


        # TODO(bart): this will be unnecessary when the derivative will be in C
        # update problem data in self._derivative_cache
        if q is not None:
            self._derivative_cache["q"] = q

        if l is not None:
            self._derivative_cache["l"] = l

        if u is not None:
            self._derivative_cache["u"] = u

        if Px is not None:
            if Px_idx.size == 0:
                self._derivative_cache["P"].data = Px
            else:
                self._derivative_cache["P"].data[Px_idx] = Px

        if Ax is not None:
            if Ax_idx.size == 0:
                self._derivative_cache["A"].data = Ax
            else:
                self._derivative_cache["A"].data[Ax_idx] = Ax

        # delete results from self._derivative_cache to prohibit
        # taking the derivative of unsolved problems
        if "results" in self._derivative_cache.keys():
            del self._derivative_cache["results"]


    def update_settings(self, **kwargs):
        """
        Update OSQP solver settings

        It is possible to change: 'max_iter', 'eps_abs', 'eps_rel',
                                  'eps_prim_inf', 'eps_dual_inf', 'rho'
                                  'alpha', 'delta', 'polish',
                                  'polish_refine_iter',
                                  'verbose', 'scaled_termination',
                                  'check_termination', 'time_limit',
        """

        # get arguments
        max_iter = kwargs.pop('max_iter', None)
        eps_abs = kwargs.pop('eps_abs', None)
        eps_rel = kwargs.pop('eps_rel', None)
        eps_prim_inf = kwargs.pop('eps_prim_inf', None)
        eps_dual_inf = kwargs.pop('eps_dual_inf', None)
        rho = kwargs.pop('rho', None)
        alpha = kwargs.pop('alpha', None)
        delta = kwargs.pop('delta', None)
        polish = kwargs.pop('polish', None)
        polish_refine_iter = kwargs.pop('polish_refine_iter', None)
        verbose = kwargs.pop('verbose', None)
        scaled_termination = kwargs.pop('scaled_termination', None)
        check_termination = kwargs.pop('check_termination', None)
        warm_start = kwargs.pop('warm_start', None)
        time_limit = kwargs.pop('time_limit', None)

        # update them
        if max_iter is not None:
            self._model.update_max_iter(max_iter)

        if eps_abs is not None:
            self._model.update_eps_abs(eps_abs)

        if eps_rel is not None:
            self._model.update_eps_rel(eps_rel)

        if eps_prim_inf is not None:
            self._model.update_eps_prim_inf(eps_prim_inf)

        if eps_dual_inf is not None:
            self._model.update_eps_dual_inf(eps_dual_inf)

        if rho is not None:
            self._model.update_rho(rho)

        if alpha is not None:
            self._model.update_alpha(alpha)

        if delta is not None:
            self._model.update_delta(delta)

        if polish is not None:
            self._model.update_polish(polish)

        if polish_refine_iter is not None:
            self._model.update_polish_refine_iter(polish_refine_iter)

        if verbose is not None:
            self._model.update_verbose(verbose)

        if scaled_termination is not None:
            self._model.update_scaled_termination(scaled_termination)

        if check_termination is not None:
            self._model.update_check_termination(check_termination)

        if warm_start is not None:
            self._model.update_warm_start(warm_start)

        if time_limit is not None:
            self._model.update_time_limit(time_limit)

        if max_iter is None and \
           eps_abs is None and \
           eps_rel is None and \
           eps_prim_inf is None and \
           eps_dual_inf is None and \
           rho is None and \
           alpha is None and \
           delta is None and \
           polish is None and \
           polish_refine_iter is None and \
           verbose is None and \
           scaled_termination is None and \
           check_termination is None and \
           warm_start is None:
            raise ValueError("No updatable settings has been specified!")

    def solve(self):
        """
        Solve QP Problem
        """
        # Solve QP
        results = self._model.solve()

        # TODO(bart): this will be unnecessary when the derivative will be in C
        self._derivative_cache['results'] = results

        return results

    def warm_start(self, x=None, y=None):
        """
        Warm start primal or dual variables
        """
        # get problem dimensions
        (n, m) = self._model.dimensions()

        if x is not None:
            if len(x) != n:
                raise ValueError("Wrong dimension for variable x")

            if y is None:
                self._model.warm_start_x(x)

        if y is not None:
            if len(y) != m:
                raise ValueError("Wrong dimension for variable y")

            if x is None:
                self._model.warm_start_y(y)

        if x is not None and y is not None:
            self._model.warm_start(x, y)

        if x is None and y is None:
            raise ValueError("Unrecognized fields")

    def codegen(self, folder, project_type='', parameters='vectors',
                python_ext_name='emosqp', force_rewrite=False,
                FLOAT=False, LONG=True):
        """
        Generate embeddable C code for the problem
        """

        # Check parameters arguments
        if parameters == 'vectors':
            embedded = 1
        elif parameters == 'matrices':
            embedded = 2
        else:
            raise ValueError("Unknown value of 'parameters' argument.")

        # Set float and long flags
        if FLOAT:
            float_flag = 'ON'
        else:
            float_flag = 'OFF'
        if LONG:
            long_flag = 'ON'
        else:
            long_flag = 'OFF'

        # Check project_type argument
        expectedProject = ('', 'Makefile', 'MinGW Makefiles',
                           'Unix Makefiles', 'CodeBlocks', 'Xcode')
        if project_type not in expectedProject:
            raise ValueError("Unknown value of 'project_type' argument.")

        if project_type == 'Makefile':
            if system() == 'Windows':
                project_type = 'MinGW Makefiles'
            elif system() == 'Linux' or system() == 'Darwin':
                project_type = 'Unix Makefiles'

        # Convert workspace to Python
        sys.stdout.write("Getting workspace from OSQP object... \t\t\t\t")
        sys.stdout.flush()
        work = self._model._get_workspace()
        print("[done]")

        # Generate code with codegen module
        cg.codegen(work, folder, python_ext_name, project_type,
                   embedded, force_rewrite, float_flag, long_flag)

    def adjoint_derivative(self, dx=None, dy_u=None, dy_l=None,
            P_idx=None, A_idx=None, diff_mode='lsqr'):
        """
        Compute adjoint derivative after solve.
        """

        P, q = self._derivative_cache['P'], self._derivative_cache['q']
        A = self._derivative_cache['A']
        l, u = self._derivative_cache['l'], self._derivative_cache['u']

        try:
            results = self._derivative_cache['results']
        except KeyError:
            raise ValueError("Problem has not been solved. You cannot take derivatives. Please call the solve function.")

        if results.info.status != "solved":
            raise ValueError("Problem has not been solved to optimality. You cannot take derivatives")

        m, n = A.shape
        x = results.x
        y = results.y
        z = A.dot(x)
        y_u = np.maximum(y, 0)
        y_l = -np.minimum(y, 0)

        if A_idx is None:
            A_idx = A.nonzero()

        if P_idx is None:
            P_idx = P.nonzero()

        if dy_u is None:
            dy_u = np.zeros(m)
        if dy_l is None:
            dy_l = np.zeros(m)

        if 'active' not in diff_mode:
            # Make sure M matrix exists
            if 'M' not in self._derivative_cache:

                    M = spa.bmat([
                        [P, A.T @ spa.diags(y_u), -A.T @ spa.diags(y_l)],
                        [spa.diags(y_u) @ A, spa.diags(A @ x - u), None],
                        [-spa.diags(y_l) @ A, None, spa.diags(l - A @ x)]
                    ]).tocsc()
                    self._derivative_cache['M'] = M

            # Prepare rhs
            d_sol = np.concatenate([dx, dy_u, dy_l])

            # Normalized version
            d_sol = np.concatenate([dx,
                                    spa.diags(y_u) @ dy_u,
                                    spa.diags(y_l) @ dy_l])


            if diff_mode == 'lsqr':
                r_sol = - sla.lsqr(self._derivative_cache['M'].T, d_sol)[0]
            elif diff_mode == 'lu':
                r_sol = - sla.spsolve(self._derivative_cache['M'].T, d_sol)
            elif diff_mode == 'ldl':
                factor = cholesky(self._derivative_cache['M'].T.tocsc())
                r_sol = - factor(d_sol)
            elif diff_mode == 'qr':
                # TODO: Add something like https://github.com/oxfordcontrol/osqpth/pull/5
                # but use slack variables too
                raise NotImplementedError
            else:
                raise RuntimeError("Unrecognized differentiation mode: {}".format(diff_mode))

            r_x, r_yu, r_yl = np.split(r_sol, [n, n+m])

            #  print("r_yu = ", r_yu)
            #  print("r_yl = ", r_yl)

            #  # Restore normalization
            #  r_yu = spa.diags(y_u) @ r_yu
            #  r_yl = spa.diags(y_l) @ r_yl

            # Extract derivatives for the constraints
            rows, cols = A_idx
            dA_vals = (y_u[rows] - y_l[rows]) * r_x[cols] + \
                (y_u[rows] * r_yu[rows] - y_l[rows] * r_yl[rows]) * x[cols]
            dA = spa.csc_matrix((dA_vals, (rows, cols)), shape=A.shape)
            du = - y_u * r_yu
            dl = y_l * r_yl

        elif diff_mode == 'lu_active':
            # Taken from https://github.com/oxfordcontrol/osqp-python/blob/0363d028b2321017049360d2eb3c0cf206028c43/modulepurepy/_osqp.py#L1717
            # Guess which linear constraints are lower-active, upper-active, free
            ind_low = np.where(z - l < - y)[0]
            ind_upp = np.where(u - z < y)[0]
            n_low = len(ind_low)
            n_upp = len(ind_upp)

            # Form A_red from the assumed active constraints
            A_red = spa.vstack([A[ind_low], A[ind_upp]])

            # Form reduced dy
            dy_red = np.concatenate([dy_l, dy_u])[np.concatenate([ind_low, ind_upp])]

            # Form KKT linear system
            KKT = spa.vstack([spa.hstack([P, A_red.T]),
                            spa.hstack([A_red, spa.csc_matrix((n_low + n_upp, n_low + n_upp))])], format='csc')
            rhs = - np.hstack([dx, dy_red])

            # Get solution
            r_sol = sla.spsolve(KKT, rhs)
            r_x, r_yl, r_yu =  np.split(r_sol, [n, n + n_low])

            r_y = np.zeros(m)
            r_y[ind_low] = r_yl
            r_y[ind_upp] = r_yu


            # Extract derivatives for the constraints A, l, u
            dl = - np.hstack([r_yl[np.where(ind_low == j)[0]] if j in ind_low else 0
                for j in range(m)])
            du = - np.hstack([r_yu[np.where(ind_upp == j)[0]] if j in ind_upp else 0
                for j in range(m)])
            rows, cols = A_idx
            dA_vals = y[rows] * r_x[cols] + r_y[rows] * x[cols]
            dA = spa.csc_matrix((dA_vals, (rows, cols)), shape=A.shape)

        else:
            raise RuntimeError("Unrecognized differentiation mode: {}".format(diff_mode))

        # Extract derivatives for the cost (P, q)
        rows, cols = P_idx
        dP_vals = .5 * (r_x[rows] * x[cols] + r_x[cols] * x[rows])
        dP = spa.csc_matrix((dP_vals, P_idx), shape=P.shape)
        dq = r_x

        return (dP, dq, dA, dl, du)
