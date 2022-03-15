from types import SimpleNamespace
import osqp
from osqp.tests.utils import load_high_accuracy, rel_tol, abs_tol, decimal_tol, Random, SOLVER_TYPES
import numpy as np
import scipy as sp
from scipy import sparse

import pytest
import numpy.testing as nptest


@pytest.fixture(params=SOLVER_TYPES)
def self(request):
    self = SimpleNamespace()

    sp.random.seed(1)

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
    self.opts = {'eps_abs': 1e-08,
                 'eps_rel': 1e-08,
                 'verbose': False}
    self.model = osqp.OSQP()
    self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
                     **self.opts)

    return self


def test_solve(self):
    # Solve problem
    res = self.model.solve()

    # Assert close
    x_sol, y_sol, obj_sol = load_high_accuracy('test_solve')
    # Assert close
    nptest.assert_allclose(res.x, x_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_allclose(res.y, y_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_almost_equal(
        res.info.obj_val, obj_sol, decimal=decimal_tol)


def test_update_P(self):
    # Update matrix P
    Px = self.P_triu_new.data
    Px_idx = np.arange(self.P_triu_new.nnz)
    self.model.update(Px=Px, Px_idx=Px_idx)
    res = self.model.solve()

    x_sol, y_sol, obj_sol = load_high_accuracy('test_update_P')
    # Assert close
    nptest.assert_allclose(res.x, x_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_allclose(res.y, y_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_almost_equal(
        res.info.obj_val, obj_sol, decimal=decimal_tol)


def test_update_P_partial(self):
    with Random(4234):
        n_changed = np.random.randint(self.P_triu.nnz)
        changed_data = np.random.random(n_changed)
        changed_indices = np.random.choice(np.arange(self.P_triu.nnz), n_changed)
    changed_P_triu_data = self.P_triu.data.copy()
    changed_P_triu_data[changed_indices] = changed_data
    changed_P_triu = np.array(sparse.coo_matrix((changed_P_triu_data, (self.P_triu.row, self.P_triu.col))).todense())
    changed_P = sparse.csc_matrix(np.triu(changed_P_triu, 1) + np.tril(changed_P_triu.T))
    if not np.all(np.linalg.eigvals(changed_P.todense()) > 0):
        pytest.skip("Perturbed P not positive semi-definite")

    self.model.update(Px=changed_data, Px_idx=changed_indices)
    res1 = self.model.solve()

    # The results we obtain should be the same as if we were solving a new problem with the new P
    model = osqp.OSQP()
    model.setup(P=changed_P, q=self.q, A=self.A, l=self.l, u=self.u,
                     **self.opts)
    res2 = model.solve()

    assert np.allclose(res1.x, res2.x, atol=abs_tol, rtol=rel_tol)
    assert np.allclose(res1.y, res2.y, atol=abs_tol, rtol=rel_tol)
    assert np.allclose(res1.info.obj_val, res2.info.obj_val, atol=abs_tol, rtol=rel_tol)


def test_update_A_partial(self):
    with Random(60023):
        n_changed = np.random.randint(self.A.nnz)
        changed_data = np.random.random(n_changed)
        changed_indices = np.random.choice(np.arange(self.A.nnz), n_changed)
    changed_A_data = self.A.data.copy()
    changed_A_data[changed_indices] = changed_data
    changed_A = sparse.csc_matrix((changed_A_data, self.A.indices, self.A.indptr))

    self.model.update(Ax=changed_data, Ax_idx=changed_indices)
    res1 = self.model.solve()

    # The results we obtain should be the same as if we were solving a new problem with the new A
    model = osqp.OSQP()
    model.setup(P=self.P, q=self.q, A=changed_A, l=self.l, u=self.u,
                     **self.opts)
    res2 = model.solve()

    assert np.allclose(res1.x, res2.x, atol=abs_tol, rtol=rel_tol)
    assert np.allclose(res1.y, res2.y, atol=abs_tol, rtol=rel_tol)
    assert np.allclose(res1.info.obj_val, res2.info.obj_val, atol=abs_tol, rtol=rel_tol)


def test_update_P_A_partial(self):
    with Random(54355):
        n_P_changed = np.random.randint(self.P_triu.nnz)
        _changed_P_data = np.random.random(n_P_changed)
        changed_P_indices = np.random.choice(np.arange(self.P_triu.nnz), n_P_changed)
        n_A_changed = np.random.randint(self.A.nnz)
        _changed_A_data = np.random.random(n_A_changed)
        changed_A_indices = np.random.choice(np.arange(self.A.nnz), n_A_changed)

    changed_P_triu_data = self.P_triu.data.copy()
    changed_P_triu_data[changed_P_indices] = _changed_P_data
    changed_P_triu = np.array(sparse.coo_matrix((changed_P_triu_data, (self.P_triu.row, self.P_triu.col))).todense())
    changed_P = sparse.csc_matrix(np.triu(changed_P_triu, 1) + np.tril(changed_P_triu.T))
    if not np.all(np.linalg.eigvals(changed_P.todense()) > 0):
        pytest.skip("Perturbed P not positive semi-definite")

    changed_A_data = self.A.data.copy()
    changed_A_data[changed_A_indices] = _changed_A_data
    changed_A = sparse.csc_matrix((changed_A_data, self.A.indices, self.A.indptr))

    self.model.update(Px=_changed_P_data, Px_idx=changed_P_indices, Ax=_changed_A_data, Ax_idx=changed_A_indices)
    res1 = self.model.solve()

    # The results we obtain should be the same as if we were solving a new problem with the new P/A
    model = osqp.OSQP()
    model.setup(P=changed_P, q=self.q, A=changed_A, l=self.l, u=self.u,
                     **self.opts)
    res2 = model.solve()

    assert np.allclose(res1.x, res2.x, atol=abs_tol, rtol=rel_tol)
    assert np.allclose(res1.y, res2.y, atol=abs_tol, rtol=rel_tol)
    assert np.allclose(res1.info.obj_val, res2.info.obj_val, atol=abs_tol, rtol=rel_tol)


def test_update_P_allind(self):
    # Update matrix P
    Px = self.P_triu_new.data
    self.model.update(Px=Px)
    res = self.model.solve()

    x_sol, y_sol, obj_sol = load_high_accuracy('test_update_P_allind')
    # Assert close
    nptest.assert_allclose(res.x, x_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_allclose(res.y, y_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_almost_equal(
        res.info.obj_val, obj_sol, decimal=decimal_tol)


def test_update_A(self):
    # Update matrix A
    Ax = self.A_new.data
    Ax_idx = np.arange(self.A_new.nnz)
    self.model.update(Ax=Ax, Ax_idx=Ax_idx)
    res = self.model.solve()

    x_sol, y_sol, obj_sol = load_high_accuracy('test_update_A')
    # Assert close
    nptest.assert_allclose(res.x, x_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_allclose(res.y, y_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_almost_equal(
        res.info.obj_val, obj_sol, decimal=decimal_tol)


def test_update_A_allind(self):
    # Update matrix A
    Ax = self.A_new.data
    self.model.update(Ax=Ax)
    res = self.model.solve()

    x_sol, y_sol, obj_sol = load_high_accuracy('test_update_A_allind')
    # Assert close
    nptest.assert_allclose(res.x, x_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_allclose(res.y, y_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_almost_equal(
        res.info.obj_val, obj_sol, decimal=decimal_tol)


def test_update_P_A_indP_indA(self):
    # Update matrices P and A
    Px = self.P_triu_new.data
    Px_idx = np.arange(self.P_triu_new.nnz)
    Ax = self.A_new.data
    Ax_idx = np.arange(self.A_new.nnz)
    self.model.update(Px=Px, Px_idx=Px_idx, Ax=Ax, Ax_idx=Ax_idx)
    res = self.model.solve()

    x_sol, y_sol, obj_sol = load_high_accuracy('test_update_P_A_indP_indA')
    # Assert close
    nptest.assert_allclose(res.x, x_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_allclose(res.y, y_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_almost_equal(
        res.info.obj_val, obj_sol, decimal=decimal_tol)


def test_update_P_A_indP(self):
    # Update matrices P and A
    Px = self.P_triu_new.data
    Px_idx = np.arange(self.P_triu_new.nnz)
    Ax = self.A_new.data
    self.model.update(Px=Px, Px_idx=Px_idx, Ax=Ax)
    res = self.model.solve()

    x_sol, y_sol, obj_sol = load_high_accuracy('test_update_P_A_indP')
    # Assert close
    nptest.assert_allclose(res.x, x_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_allclose(res.y, y_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_almost_equal(
        res.info.obj_val, obj_sol, decimal=decimal_tol)


def test_update_P_A_indA(self):
    # Update matrices P and A
    Px = self.P_triu_new.data
    Ax = self.A_new.data
    Ax_idx = np.arange(self.A_new.nnz)
    self.model.update(Px=Px, Ax=Ax, Ax_idx=Ax_idx)
    res = self.model.solve()

    x_sol, y_sol, obj_sol = load_high_accuracy('test_update_P_A_indA')
    # Assert close
    nptest.assert_allclose(res.x, x_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_allclose(res.y, y_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_almost_equal(
        res.info.obj_val, obj_sol, decimal=decimal_tol)


def test_update_P_A_allind(self):
    # Update matrices P and A
    Px = self.P_triu_new.data
    Ax = self.A_new.data
    self.model.update(Px=Px, Ax=Ax)
    res = self.model.solve()

    x_sol, y_sol, obj_sol = load_high_accuracy('test_update_P_A_allind')
    # Assert close
    nptest.assert_allclose(res.x, x_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_allclose(res.y, y_sol, rtol=rel_tol, atol=abs_tol)
    nptest.assert_almost_equal(
        res.info.obj_val, obj_sol, decimal=decimal_tol)