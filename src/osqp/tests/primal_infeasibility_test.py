from types import SimpleNamespace
import osqp
from osqp import constant
from osqp.tests.utils import SOLVER_TYPES
from scipy import sparse
import numpy as np
import pytest


@pytest.fixture(params=SOLVER_TYPES)
def self(request):
    self = SimpleNamespace()
    self.opts = {
        'verbose': False,
        'eps_abs': 1e-05,
        'eps_rel': 1e-05,
        'eps_dual_inf': 1e-20,
        'max_iter': 2500,
        'polish': False,
    }
    self.model = osqp.OSQP()
    self.model.solver_type = request.param
    return self


def test_primal_infeasible_problem(self):

    # Simple QP problem
    np.random.seed(4)

    self.n = 50
    self.m = 500
    # Generate random Matrices
    Pt = np.random.rand(self.n, self.n)
    self.P = sparse.triu(Pt.T.dot(Pt), format='csc')
    self.q = np.random.rand(self.n)
    self.A = sparse.random(self.m, self.n).tolil()  # Lil for efficiency
    self.u = 3 + np.random.randn(self.m)
    self.l = -3 + np.random.randn(self.m)

    # Make random problem primal infeasible
    self.A[int(self.n / 2), :] = self.A[int(self.n / 2) + 1, :]
    self.l[int(self.n / 2)] = self.u[int(self.n / 2) + 1] + 10 * np.random.rand()
    self.u[int(self.n / 2)] = self.l[int(self.n / 2)] + 0.5

    # Convert A to csc
    self.A = self.A.tocsc()

    self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, **self.opts)

    # Solve problem with OSQP
    res = self.model.solve()

    assert res.info.status_val == constant('OSQP_PRIMAL_INFEASIBLE')


def test_primal_and_dual_infeasible_problem(self):

    self.n = 2
    self.m = 4
    self.P = sparse.csc_matrix((2, 2))
    self.q = np.array([-1.0, -1.0])
    self.A = sparse.csc_matrix([[1.0, -1.0], [-1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    self.l = np.array([1.0, 1.0, 0.0, 0.0])
    self.u = np.inf * np.ones(self.m)

    self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, **self.opts)

    res = self.model.solve()

    assert res.info.status_val in (constant('OSQP_PRIMAL_INFEASIBLE'), constant('OSQP_DUAL_INFEASIBLE'))
