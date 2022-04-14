# Test osqp python module
import osqp
# import osqppurepy as osqp
import numpy as np
from scipy import sparse

# Unit Test
import unittest
import pytest
import numpy.testing as nptest
import shutil as sh
import sys


class codegen_vectors_tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        P = sparse.diags([11., 0.], format='csc')
        q = np.array([3, 4])
        A = sparse.csc_matrix(
            [[-1, 0], [0, -1], [-1, -3], [2, 5], [3, 4]])
        u = np.array([0, 0, -15, 100, 80])
        l = -np.inf * np.ones(len(u))
        n = P.shape[0]
        m = A.shape[0]
        opts = {'verbose': False,
                     'eps_abs': 1e-08,
                     'eps_rel': 1e-08,
                     'rho': 0.01,
                     'alpha': 1.6,
                     'max_iter': 10000,
                     'warm_start': True}

        model = osqp.OSQP()
        model.setup(P=P, q=q, A=A, l=l, u=u, **opts)

        model_dir = model.codegen('code', python_ext_name='vec_emosqp',
                           force_rewrite=True)
        sh.rmtree('code')
        sys.path.append(model_dir)

        cls.m = m
        cls.n = n
        cls.P = P
        cls.q = q
        cls.A = A
        cls.l = l
        cls.u = u
        cls.opts = opts

    def setUp(self):
        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
                         **self.opts)

    def test_solve(self):
        import vec_emosqp

        # Solve problem
        x, y, _, _, _ = vec_emosqp.solve()

        # Assert close
        nptest.assert_array_almost_equal(x, np.array([0., 5.]), decimal=5)
        nptest.assert_array_almost_equal(
            y, np.array([1.66666667, 0., 1.33333333, 0., 0.]), decimal=5)

    def test_update_q(self):
        import vec_emosqp

        # Update linear cost and solve the problem
        q_new = np.array([10., 20.])
        vec_emosqp.update_lin_cost(q_new)
        x, y, _, _, _ = vec_emosqp.solve()

        # Assert close
        nptest.assert_array_almost_equal(x, np.array([0., 5.]), decimal=5)
        nptest.assert_array_almost_equal(
            y, np.array([3.33333334, 0., 6.66666667, 0., 0.]), decimal=5)

        # Update linear cost to the original value
        vec_emosqp.update_lin_cost(self.q)

    def test_update_l(self):
        import vec_emosqp

        # Update lower bound
        l_new = -100. * np.ones(self.m)
        vec_emosqp.update_lower_bound(l_new)
        x, y, _, _, _ = vec_emosqp.solve()

        # Assert close
        nptest.assert_array_almost_equal(x, np.array([0., 5.]), decimal=5)
        nptest.assert_array_almost_equal(
            y, np.array([1.66666667, 0., 1.33333333, 0., 0.]), decimal=5)

        # Update lower bound to the original value
        vec_emosqp.update_lower_bound(self.l)

    def test_update_u(self):
        import vec_emosqp

        # Update upper bound
        u_new = 1000. * np.ones(self.m)
        vec_emosqp.update_upper_bound(u_new)
        x, y, _, _, _ = vec_emosqp.solve()

        # Assert close
        nptest.assert_array_almost_equal(
            x, np.array([-1.51515152e-01, -3.33282828e+02]), decimal=4)
        nptest.assert_array_almost_equal(
            y, np.array([0., 0., 1.33333333, 0., 0.]), decimal=4)

        # Update upper bound to the original value
        vec_emosqp.update_upper_bound(self.u)

    def test_update_bounds(self):
        import vec_emosqp

        # Update upper bound
        l_new = -100. * np.ones(self.m)
        u_new = 1000. * np.ones(self.m)
        vec_emosqp.update_bounds(l_new, u_new)
        x, y, _, _, _ = vec_emosqp.solve()

        # Assert close
        nptest.assert_array_almost_equal(
            x, np.array([-0.12727273, -19.94909091]), decimal=4)
        nptest.assert_array_almost_equal(
            y, np.array([0., 0., 0., -0.8, 0.]), decimal=4)

        # Update upper bound to the original value
        vec_emosqp.update_bounds(self.l, self.u)
