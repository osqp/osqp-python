# Test osqp python module
import osqp
# import osqppurepy as osqp
import numpy as np
from scipy import sparse

# Unit Test
import unittest
import numpy.testing as nptest

class non_convex_tests(unittest.TestCase):

    def setUp(self):

        # Simple QP problem
        self.P = sparse.csc_matrix([[2., 5.], [5., 1.]])
        self.q = np.array([3, 4])
        self.A = sparse.csc_matrix([[-1.0, 0.], [0., -1.], [-1., 3.], [2., 5.], [3., 4]])
        self.u = np.array([0., 0., -15, 100, 80])
        self.l = -np.inf * np.ones(len(self.u))
        self.opts = {'verbose': False}
        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
                         **self.opts)

    def test_non_convex(self):
        # Solve problem
        res = self.model.solve()

        # Assert close
        self.assertEqual(res.info.status_val, self.model.constant('OSQP_NON_CVX'))
        nptest.assert_approx_equal(res.info.obj_val, self.model.constant('OSQP_NAN'))

