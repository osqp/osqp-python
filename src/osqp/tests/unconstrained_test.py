import osqp
from osqp.tests.utils import load_high_accuracy
import numpy as np
from scipy import sparse
import pytest
import numpy.testing as nptest


@pytest.fixture
def self(algebra, solver_type, atol, rtol, decimal_tol):
    np.random.seed(4)

    self.n = 30
    self.m = 0
    P = sparse.diags(np.random.rand(self.n)) + 0.2 * sparse.eye(self.n)
    self.P = P.tocsc()
    self.q = np.random.randn(self.n)
    self.A = sparse.csc_matrix((self.m, self.n))
    self.l = np.array([])
    self.u = np.array([])
    self.opts = {
        'verbose': False,
        'eps_abs': 1e-08,
        'eps_rel': 1e-08,
        'polishing': False,
    }
    self.model = osqp.OSQP(algebra=algebra)
    self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, solver_type=solver_type, **self.opts)

    self.rtol = rtol
    self.atol = atol
    self.decimal_tol = decimal_tol

    return self


def test_unconstrained_problem(self):
    # Solve problem
    res = self.model.solve()

    # Assert close
    x_sol, _, obj_sol = load_high_accuracy('test_unconstrained_problem')
    # Assert close
    nptest.assert_allclose(res.x, x_sol, rtol=self.rtol, atol=self.atol)
    nptest.assert_almost_equal(res.info.obj_val, obj_sol, decimal=self.decimal_tol)
