# Test osqp python module
import osqp
# import osqppurepy as osqp
import numpy as np
import numpy.random as npr
from scipy import sparse
import scipy as sp
import numdifftools as nd
import numpy.testing as npt

# Unit Test
import unittest


#  diff_modes = ['qr', 'lsqr', 'qr_active']
diff_modes = ['qr_active']

ATOL = 1e-2
RTOL = 1e-4

class derivative_tests(unittest.TestCase):

    def get_grads(self, n=10, m=3, P_scale=1., A_scale=1., diff_mode=diff_modes[0]):
        npr.seed(1)
        L = np.random.randn(n, n)
        P = sparse.csc_matrix(L.dot(L.T))
        x_0 = npr.randn(n)
        s_0 = npr.rand(m)
        A = sparse.csc_matrix(npr.randn(m, n))
        u = A.dot(x_0) + s_0
        l = - 10 * npr.rand(m)
        q = npr.randn(n)
        true_x = npr.randn(n)

        # Get gradients by solving with osqp
        m = osqp.OSQP()
        m.setup(P, q, A, l, u, verbose=True)
        results = m.solve()
        x = results.x
        grads = m.adjoint_derivative(dx=x - true_x, diff_mode=diff_mode)

        return [P, q,  A, l, u, true_x], grads

    def test_dl_dp(self, verbose=True):
        n, m = 5, 5
        for diff_mode in diff_modes:

            # Get gradients
            [P, q, A, l, u, true_x], [dP, dq, dA, dl, du] = self.get_grads(
                n=n, m=m, P_scale=100., A_scale=100., diff_mode=diff_mode)
            print(f'--- {diff_mode}')

            def f(q):
                m = osqp.OSQP()
                m.setup(P, q, A, l, u, verbose=False)
                res = m.solve()
                x_hat = res.x

                return 0.5 * np.sum(np.square(x_hat - true_x))

            dq_fd = nd.Gradient(f)(q)  # Take numerical gradient of f
            if verbose:
                print('dq_fd: ', np.round(dq_fd, decimals=4))
                print('dq: ', np.round(dq, decimals=4))
            npt.assert_allclose(dq_fd, dq, rtol=RTOL, atol=ATOL)
