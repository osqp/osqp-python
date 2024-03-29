from types import SimpleNamespace
import osqp
from osqp.tests.utils import load_high_accuracy
import numpy as np
from scipy import sparse
import pytest
import numpy.testing as nptest


@pytest.fixture
def self(algebra, solver_type, atol, rtol, decimal_tol):
    self = SimpleNamespace()

    np.random.seed(1)

    self.n = 5
    self.m = 8
    p = 0.7

    Pt = sparse.random(self.n, self.n, density=p)
    Pt_new = Pt.copy()
    Pt_new.data += 0.1 * np.random.randn(Pt.nnz)

    self.P = (Pt.T.dot(Pt) + sparse.eye(self.n)).tocsc()
    self.P_new = (Pt_new.T.dot(Pt_new) + sparse.eye(self.n)).tocsc()
    self.P_triu = sparse.triu(self.P)
    self.P_triu_new = sparse.triu(self.P_new)
    self.q = np.random.randn(self.n)
    self.A = sparse.random(self.m, self.n, density=p, format='csc')
    self.A_new = self.A.copy()
    self.A_new.data += np.random.randn(self.A_new.nnz)
    self.l = np.zeros(self.m)
    self.u = 30 + np.random.randn(self.m)
    self.opts = {'eps_abs': 1e-08, 'eps_rel': 1e-08, 'verbose': False}
    self.model = osqp.OSQP(algebra=algebra)
    self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, solver_type=solver_type, **self.opts)

    self.rtol = rtol
    self.atol = atol
    self.decimal_tol = decimal_tol

    return self


def test_solve(self):
    res = self.model.solve()

    x_sol, y_sol, obj_sol = load_high_accuracy('test_solve')
    nptest.assert_allclose(res.x, x_sol, rtol=self.rtol, atol=self.atol)
    nptest.assert_allclose(res.y, y_sol, rtol=self.rtol, atol=self.atol)
    nptest.assert_almost_equal(res.info.obj_val, obj_sol, decimal=self.decimal_tol)


def test_update_P(self):
    # Update matrix P
    Px = self.P_triu_new.data
    Px_idx = np.arange(self.P_triu_new.nnz)
    self.model.update(Px=Px, Px_idx=Px_idx)
    res = self.model.solve()

    x_sol, y_sol, obj_sol = load_high_accuracy('test_update_P')

    if self.model.algebra != 'cuda':  # pytest-todo
        nptest.assert_allclose(res.x, x_sol, rtol=self.rtol, atol=self.atol)
        nptest.assert_allclose(res.y, y_sol, rtol=self.rtol, atol=self.atol)
        nptest.assert_almost_equal(res.info.obj_val, obj_sol, decimal=self.decimal_tol)


def test_update_P_allind(self):
    # Update matrix P
    Px = self.P_triu_new.data
    self.model.update(Px=Px)
    res = self.model.solve()

    x_sol, y_sol, obj_sol = load_high_accuracy('test_update_P_allind')
    # Assert close
    nptest.assert_allclose(res.x, x_sol, rtol=self.rtol, atol=self.atol)
    nptest.assert_allclose(res.y, y_sol, rtol=self.rtol, atol=self.atol)
    nptest.assert_almost_equal(res.info.obj_val, obj_sol, decimal=self.decimal_tol)


def test_update_A(self):
    # Update matrix A
    Ax = self.A_new.data
    Ax_idx = np.arange(self.A_new.nnz)
    self.model.update(Ax=Ax, Ax_idx=Ax_idx)
    res = self.model.solve()

    x_sol, y_sol, obj_sol = load_high_accuracy('test_update_A')

    if self.model.algebra != 'cuda':  # pytest-todo
        nptest.assert_allclose(res.x, x_sol, rtol=self.rtol, atol=self.atol)
        nptest.assert_allclose(res.y, y_sol, rtol=self.rtol, atol=self.atol)
        nptest.assert_almost_equal(res.info.obj_val, obj_sol, decimal=self.decimal_tol)


def test_update_A_allind(self):
    # Update matrix A
    Ax = self.A_new.data
    self.model.update(Ax=Ax)
    res = self.model.solve()

    x_sol, y_sol, obj_sol = load_high_accuracy('test_update_A_allind')
    # Assert close
    nptest.assert_allclose(res.x, x_sol, rtol=self.rtol, atol=self.atol)
    nptest.assert_allclose(res.y, y_sol, rtol=self.rtol, atol=self.atol)
    nptest.assert_almost_equal(res.info.obj_val, obj_sol, decimal=self.decimal_tol)


def test_update_P_A_indP_indA(self):
    # Update matrices P and A
    Px = self.P_triu_new.data
    Px_idx = np.arange(self.P_triu_new.nnz)
    Ax = self.A_new.data
    Ax_idx = np.arange(self.A_new.nnz)
    self.model.update(Px=Px, Px_idx=Px_idx, Ax=Ax, Ax_idx=Ax_idx)
    res = self.model.solve()

    x_sol, y_sol, obj_sol = load_high_accuracy('test_update_P_A_indP_indA')

    if self.model.algebra != 'cuda':  # pytest-todo
        nptest.assert_allclose(res.x, x_sol, rtol=self.rtol, atol=self.atol)
        nptest.assert_allclose(res.y, y_sol, rtol=self.rtol, atol=self.atol)
        nptest.assert_almost_equal(res.info.obj_val, obj_sol, decimal=self.decimal_tol)


def test_update_P_A_indP(self):
    # Update matrices P and A
    Px = self.P_triu_new.data
    Px_idx = np.arange(self.P_triu_new.nnz)
    Ax = self.A_new.data
    self.model.update(Px=Px, Px_idx=Px_idx, Ax=Ax)
    res = self.model.solve()

    x_sol, y_sol, obj_sol = load_high_accuracy('test_update_P_A_indP')

    if self.model.algebra != 'cuda':  # pytest-todo
        nptest.assert_allclose(res.x, x_sol, rtol=self.rtol, atol=self.atol)
        nptest.assert_allclose(res.y, y_sol, rtol=self.rtol, atol=self.atol)
        nptest.assert_almost_equal(res.info.obj_val, obj_sol, decimal=self.decimal_tol)


def test_update_P_A_indA(self):
    # Update matrices P and A
    Px = self.P_triu_new.data
    Ax = self.A_new.data
    Ax_idx = np.arange(self.A_new.nnz)
    self.model.update(Px=Px, Ax=Ax, Ax_idx=Ax_idx)
    res = self.model.solve()

    x_sol, y_sol, obj_sol = load_high_accuracy('test_update_P_A_indA')

    if self.model.algebra != 'cuda':  # pytest-todo
        nptest.assert_allclose(res.x, x_sol, rtol=self.rtol, atol=self.atol)
        nptest.assert_allclose(res.y, y_sol, rtol=self.rtol, atol=self.atol)
        nptest.assert_almost_equal(res.info.obj_val, obj_sol, decimal=self.decimal_tol)


def test_update_P_A_allind(self):
    # Update matrices P and A
    Px = self.P_triu_new.data
    Ax = self.A_new.data
    self.model.update(Px=Px, Ax=Ax)
    res = self.model.solve()

    x_sol, y_sol, obj_sol = load_high_accuracy('test_update_P_A_allind')

    nptest.assert_allclose(res.x, x_sol, rtol=self.rtol, atol=self.atol)
    nptest.assert_allclose(res.y, y_sol, rtol=self.rtol, atol=self.atol)
    nptest.assert_almost_equal(res.info.obj_val, obj_sol, decimal=self.decimal_tol)
