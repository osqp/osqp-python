from types import SimpleNamespace
import pytest
from scipy import sparse
import numpy as np
import numpy.testing as nptest
from osqp import OSQP
from osqp.tests.utils import load_high_accuracy


@pytest.fixture
def self(algebra, solver_type, atol, rtol, decimal_tol):
    ns = SimpleNamespace()
    ns.P = sparse.diags([11.0, 0.0], format='csc')
    ns.q = np.array([3, 4])
    ns.A = sparse.csc_matrix([[-1, 0], [0, -1], [-1, -3], [2, 5], [3, 4]])
    ns.u = np.array([0.0, 0.0, -15, 100, 80])
    ns.l = -1e06 * np.ones(len(ns.u))
    ns.n = ns.P.shape[0]
    ns.m = ns.A.shape[0]
    ns.opts = {
        'verbose': False,
        'eps_abs': 1e-09,
        'eps_rel': 1e-09,
        'max_iter': 2500,
        'rho': 0.1,
        'adaptive_rho': False,
        'polishing': False,
        'check_termination': 1,
        'warm_starting': True,
        'solver_type': solver_type,
    }
    ns.model = OSQP(algebra=algebra)
    ns.model.setup(P=ns.P, q=ns.q, A=ns.A, l=ns.l, u=ns.u, **ns.opts)
    ns.atol = atol
    ns.rtol = rtol
    ns.decimal_tol = decimal_tol
    return ns


def test_basic_QP(self):
    res = self.model.solve()

    x_sol, y_sol, obj_sol = load_high_accuracy('test_basic_QP')
    nptest.assert_allclose(res.x, x_sol, rtol=self.rtol, atol=self.atol)
    nptest.assert_allclose(res.y, y_sol, rtol=self.rtol, atol=self.atol)
    nptest.assert_almost_equal(res.info.obj_val, obj_sol, decimal=self.decimal_tol)


def test_update_q(self):
    # Update linear cost
    q_new = np.array([10, 20])
    self.model.update(q=q_new)
    res = self.model.solve()

    x_sol, y_sol, obj_sol = load_high_accuracy('test_update_q')
    nptest.assert_allclose(res.x, x_sol, rtol=self.rtol, atol=self.atol)
    nptest.assert_allclose(res.y, y_sol, rtol=self.rtol, atol=self.atol)
    nptest.assert_almost_equal(res.info.obj_val, obj_sol, decimal=self.decimal_tol)


def test_update_l(self):
    # Update lower bound
    l_new = -50 * np.ones(self.m)
    self.model.update(l=l_new)
    res = self.model.solve()

    x_sol, y_sol, obj_sol = load_high_accuracy('test_update_l')
    nptest.assert_allclose(res.x, x_sol, rtol=self.rtol, atol=self.atol)
    nptest.assert_allclose(res.y, y_sol, rtol=self.rtol, atol=self.atol)
    nptest.assert_almost_equal(res.info.obj_val, obj_sol, decimal=self.decimal_tol)


def test_update_u(self):
    # Update lower bound
    u_new = 1000 * np.ones(self.m)
    self.model.update(u=u_new)
    res = self.model.solve()

    x_sol, y_sol, obj_sol = load_high_accuracy('test_update_u')
    nptest.assert_allclose(res.x, x_sol, rtol=self.rtol, atol=self.atol)
    nptest.assert_allclose(res.y, y_sol, rtol=self.rtol, atol=self.atol)
    nptest.assert_almost_equal(res.info.obj_val, obj_sol, decimal=self.decimal_tol)


def test_update_bounds(self):
    # Update lower bound
    l_new = -100 * np.ones(self.m)
    # Update lower bound
    u_new = 1000 * np.ones(self.m)
    self.model.update(u=u_new, l=l_new)
    res = self.model.solve()

    x_sol, y_sol, obj_sol = load_high_accuracy('test_update_bounds')
    if self.model.algebra != 'cuda':  # pytest-todo
        nptest.assert_allclose(res.x, x_sol, rtol=self.rtol, atol=self.atol)
        nptest.assert_allclose(res.y, y_sol, rtol=self.rtol, atol=self.atol)
        nptest.assert_almost_equal(res.info.obj_val, obj_sol, decimal=self.decimal_tol)
    else:
        assert res.info.status_val == self.model.constant('OSQP_PRIMAL_INFEASIBLE')


def test_update_max_iter(self):
    self.model.update_settings(max_iter=80)
    res = self.model.solve()

    assert res.info.status_val == self.model.constant('OSQP_MAX_ITER_REACHED')


def test_update_check_termination(self):
    self.model.update_settings(check_termination=0)
    res = self.model.solve()

    assert res.info.iter == self.opts['max_iter']


def test_update_rho(self):
    res_default = self.model.solve()

    # Setup with different rho and update
    default_opts = self.opts.copy()
    default_opts['rho'] = 0.7
    model = OSQP(algebra=self.model.algebra)
    model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, **default_opts)
    model.update_settings(rho=self.opts['rho'])
    res_updated_rho = model.solve()

    # Assert same number of iterations
    assert res_default.info.iter == res_updated_rho.info.iter


def test_upper_triangular_P(self):
    res_default = self.model.solve()

    # Get upper triangular P
    P_triu = sparse.triu(self.P, format='csc')

    # Setup and solve with upper triangular part only
    model = OSQP(algebra=self.model.algebra)
    model.setup(P=P_triu, q=self.q, A=self.A, l=self.l, u=self.u, **self.opts)
    res_triu = model.solve()

    nptest.assert_allclose(res_default.x, res_triu.x, rtol=self.rtol, atol=self.atol)
    nptest.assert_allclose(res_default.y, res_triu.y, rtol=self.rtol, atol=self.atol)
    nptest.assert_almost_equal(
        res_default.info.obj_val,
        res_triu.info.obj_val,
        decimal=self.decimal_tol,
    )


def test_update_invalid(self):
    # can't update unsupported setting
    with pytest.raises(ValueError):
        self.model.update_settings(foo=42)
