from types import SimpleNamespace
import osqp
import numpy as np
from scipy import sparse
import pytest
import numpy.testing as nptest
from osqp.tests.utils import load_high_accuracy


@pytest.fixture
def self(algebra, solver_type, atol, rtol, decimal_tol):
    self = SimpleNamespace()

    np.random.seed(4)

    self.n = 30
    self.m = 30
    self.P = sparse.csc_matrix((self.n, self.n))
    self.q = np.zeros(self.n)
    self.A = sparse.random(self.m, self.n, density=1.0, format='csc')
    self.u = np.random.rand(self.m)
    self.l = self.u
    self.opts = {
        'verbose': False,
        'eps_abs': 1e-06,
        'eps_rel': 1e-06,
        'scaling': True,
        'alpha': 1.6,
        'max_iter': 5000,
        'polishing': False,
        'warm_starting': True,
        'polish_refine_iter': 4,
        'solver_type': solver_type,
    }

    self.model = osqp.OSQP(algebra=algebra)
    self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, **self.opts)

    self.rtol = rtol
    self.atol = atol
    self.decimal_tol = decimal_tol

    return self


def test_feasibility_problem(self):
    res = self.model.solve()

    x_sol, y_sol, obj_sol = load_high_accuracy('test_feasibility_problem')

    if self.model.solver_type == 'direct':  # pytest-todo
        nptest.assert_allclose(res.x, x_sol, rtol=self.rtol, atol=self.atol)
        nptest.assert_allclose(res.y, y_sol, rtol=self.rtol, atol=self.atol)
        nptest.assert_almost_equal(res.info.obj_val, obj_sol, decimal=self.decimal_tol)
    else:
        assert res.info.status_val == self.model.constant('OSQP_MAX_ITER_REACHED')
