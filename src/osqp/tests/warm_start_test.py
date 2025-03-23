from types import SimpleNamespace
import numpy as np
from scipy import sparse
import pytest
import osqp


@pytest.fixture
def self(algebra, solver_type, atol, rtol, decimal_tol):
    ns = SimpleNamespace()
    ns.opts = {
        'verbose': True,
        'adaptive_rho': False,
        'eps_abs': 1e-08 if solver_type == 'direct' else 1e-2,
        'eps_rel': 1e-08 if solver_type == 'direct' else 1e-2,
        'polishing': False,
        'check_termination': 1,
        'solver_type': solver_type,
    }

    ns.model = osqp.OSQP(algebra=algebra)
    return ns


def test_warm_start(self):
    # Big problem
    np.random.seed(2)
    self.n = 100
    self.m = 200
    self.A = sparse.random(self.m, self.n, density=0.9, format='csc')
    self.l = -np.random.rand(self.m) * 2.0
    self.u = np.random.rand(self.m) * 2.0

    P = sparse.random(self.n, self.n, density=0.9)
    self.P = sparse.triu(P.dot(P.T), format='csc')
    self.q = np.random.randn(self.n)

    # Setup solver
    self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, **self.opts)

    # Solve problem with OSQP
    res = self.model.solve()

    # Store optimal values
    x_opt = res.x
    y_opt = res.y
    tot_iter = res.info.iter

    # Warm start with zeros and check if number of iterations is the same
    self.model.warm_start(x=np.zeros(self.n), y=np.zeros(self.m))
    res = self.model.solve()
    assert res.info.iter == tot_iter

    # Warm start with optimal values and check that number of iter < 10
    self.model.warm_start(x=x_opt, y=y_opt)
    res = self.model.solve()
    assert res.info.iter < 10
