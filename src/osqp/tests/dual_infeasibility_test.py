from types import SimpleNamespace
import osqp
import numpy as np
from scipy import sparse

import pytest


@pytest.fixture
def self(algebra, solver_type, atol, rtol, decimal_tol):
    ns = SimpleNamespace()
    ns.opts = {
        'verbose': False,
        'eps_abs': 1e-05,
        'eps_rel': 1e-05,
        'eps_prim_inf': 1e-15,  # Focus only on dual infeasibility
        'eps_dual_inf': 1e-6,
        'scaling': 3,
        'max_iter': 2500,
        'polishing': False,
        'check_termination': 1,
        'polish_refine_iter': 4,
        'solver_type': solver_type,
    }

    ns.model = osqp.OSQP(algebra=algebra)
    return ns


def test_dual_infeasible_lp(self):
    # Dual infeasible example
    self.P = sparse.csc_matrix((2, 2))
    self.q = np.array([2, -1])
    self.A = sparse.eye(2, format='csc')
    self.l = np.array([0.0, 0.0])
    self.u = np.array([np.inf, np.inf])

    self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, **self.opts)

    # Solve problem with OSQP
    res = self.model.solve()

    assert res.info.status_val == self.model.constant('OSQP_DUAL_INFEASIBLE')


def test_dual_infeasible_qp(self):
    # Dual infeasible example
    self.P = sparse.diags([4.0, 0.0], format='csc')
    self.q = np.array([0, 2])
    self.A = sparse.csc_matrix([[1.0, 1.0], [-1.0, 1.0]])
    self.l = np.array([-np.inf, -np.inf])
    self.u = np.array([2.0, 3.0])

    self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, **self.opts)

    # Solve problem with OSQP
    res = self.model.solve()

    assert res.info.status_val == self.model.constant('OSQP_DUAL_INFEASIBLE')


def test_primal_and_dual_infeasible_problem(self):
    self.n = 2
    self.m = 4
    self.P = sparse.csc_matrix((2, 2))
    self.q = np.array([-1.0, -1.0])
    self.A = sparse.csc_matrix([[1.0, -1.0], [-1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    self.l = np.array([1.0, 1.0, 0.0, 0.0])
    self.u = np.inf * np.ones(self.m)

    self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, **self.opts)

    # Warm start to avoid infeasibility detection at first step
    x0 = 25.0 * np.ones(self.n)
    y0 = -2.0 * np.ones(self.m)
    self.model.warm_start(x=x0, y=y0)

    # Solve
    res = self.model.solve()

    assert res.info.status_val in (
        self.model.constant('OSQP_PRIMAL_INFEASIBLE'),
        self.model.constant('OSQP_DUAL_INFEASIBLE'),
    )
