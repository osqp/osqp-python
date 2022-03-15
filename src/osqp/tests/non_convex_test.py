from types import SimpleNamespace
import osqp
from osqp import constant, default_algebra
from osqp.tests.utils import SOLVER_TYPES
import numpy as np
from scipy import sparse

import pytest
import numpy.testing as nptest


@pytest.fixture(params=SOLVER_TYPES)
def self(request):
    self = SimpleNamespace()
    self.P = sparse.triu([[2., 5.], [5., 1.]], format='csc')
    self.q = np.array([3, 4])
    self.A = sparse.csc_matrix([[-1.0, 0.], [0., -1.],
                                [-1., 3.], [2., 5.], [3., 4]])
    self.u = np.array([0., 0., -15, 100, 80])
    self.l = -np.inf * np.ones(len(self.u))
    self.model = osqp.OSQP()
    self.model.solver_type = request.param
    return self


@pytest.mark.skipif(default_algebra() not in ('legacy', 'default'), reason='Only applicable for legacy/default algebra')
def test_non_convex_small_sigma(self):
    opts = {'verbose': False, 'sigma': 1e-6}
    try:
        # Setup should fail due to (P + sigma I) having a negative
        # eigenvalue
        test_setup = 1
        self.model.setup(P=self.P, q=self.q, A=self.A,
                         l=self.l, u=self.u, **opts)
    except ValueError:
        test_setup = 0

    assert test_setup == 0


def test_non_convex_big_sigma(self):
    # Setup workspace with new sigma
    opts = {'verbose': False, 'sigma': 5}
    self.model.setup(P=self.P, q=self.q, A=self.A,
                     l=self.l, u=self.u, **opts)

    # Solve problem
    res = self.model.solve()

    assert res.info.status_val == constant('OSQP_NON_CVX')
    nptest.assert_approx_equal(res.info.obj_val, np.nan)


def test_nan(self):
    nptest.assert_approx_equal(constant('OSQP_NAN'), np.nan)
