from types import SimpleNamespace
import osqp
import numpy as np
from scipy import sparse

import pytest
import numpy.testing as nptest


@pytest.fixture
def self(algebra, solver_type, atol, rtol, decimal_tol):
    ns = SimpleNamespace()
    ns.P = sparse.triu([[2.0, 5.0], [5.0, 1.0]], format='csc')
    ns.q = np.array([3, 4])
    ns.A = sparse.csc_matrix([[-1.0, 0.0], [0.0, -1.0], [-1.0, 3.0], [2.0, 5.0], [3.0, 4]])
    ns.u = np.array([0.0, 0.0, -15, 100, 80])
    ns.l = -np.inf * np.ones(len(ns.u))
    ns.model = osqp.OSQP(algebra=algebra)
    return ns


def test_non_convex_small_sigma(self, solver_type):
    if solver_type == 'direct':
        with pytest.raises(ValueError):
            self.model.setup(
                P=self.P,
                q=self.q,
                A=self.A,
                l=self.l,
                u=self.u,
                solver_type=solver_type,
                sigma=1e-6,
            )
    else:
        self.model.setup(
            P=self.P,
            q=self.q,
            A=self.A,
            l=self.l,
            u=self.u,
            solver_type=solver_type,
            sigma=1e-6,
        )
        res = self.model.solve()

        assert res.info.status_val in (
            self.model.constant('OSQP_MAX_ITER_REACHED'),
            self.model.constant('OSQP_NON_CVX'),
        )


def test_non_convex_big_sigma(self):
    self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, sigma=5)
    res = self.model.solve()

    assert res.info.status_val == self.model.constant('OSQP_NON_CVX')


def test_nan(self):
    nptest.assert_approx_equal(self.model.constant('OSQP_NAN'), np.nan)
