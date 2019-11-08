# Test osqp python module
import osqp
# import osqppurepy as osqp
import numpy as np
import numpy.random as npr
from scipy import sparse
import numdifftools as nd
import numpy.testing as npt

# Unit Test
import unittest


#  diff_modes = ['qr', 'lsqr', 'lu_active']
diff_modes = [
              'lsqr',
              'lu_active'
              ]

ATOL = 1e-2
RTOL = 1e-4
eps_abs = 1e-06
eps_rel = 1e-06

class derivative_tests(unittest.TestCase):

    def get_grads(self, n=10, m=3, P_scale=1., A_scale=1., diff_mode=diff_modes[0]):
        npr.seed(1)
        L = np.random.randn(n, n)
        P = sparse.csc_matrix(L.dot(L.T) + 5. * sparse.eye(n))
        x_0 = npr.randn(n)
        s_0 = npr.rand(m)
        A = sparse.csc_matrix(npr.randn(m, n))
        u = A.dot(x_0) + s_0
        l = -5 - 10 * npr.rand(m)
        q = npr.randn(n)
        true_x = npr.randn(n)

        # Get gradients by solving with osqp
        m = osqp.OSQP()
        m.setup(P, q, A, l, u, eps_abs=eps_abs, eps_rel=eps_rel, verbose=False)
        results = m.solve()
        if results.info.status != "solved":
            raise ValueError("Problem not solved!")
        x = results.x
        grads = m.adjoint_derivative(dx=x - true_x, diff_mode=diff_mode)

        return [P, q,  A, l, u, true_x], grads

    def test_dl_dq(self, verbose=False):
        n, m = 5, 5
        for diff_mode in diff_modes:

            # Get gradients
            [P, q, A, l, u, true_x], [dP, dq, dA, dl, du] = self.get_grads(
                n=n, m=m, P_scale=100., A_scale=100., diff_mode=diff_mode)

            def f(q):
                m = osqp.OSQP()
                m.setup(P, q, A, l, u, eps_abs=eps_abs, eps_rel=eps_rel, verbose=False)
                res = m.solve()
                x_hat = res.x

                return 0.5 * np.sum(np.square(x_hat - true_x))

            dq_fd = nd.Gradient(f)(q)  # Take numerical gradient of f
            if verbose:
                print('--- ' + diff_mode)
                print('dq_fd: ', np.round(dq_fd, decimals=4))
                print('dq: ', np.round(dq, decimals=4))
            npt.assert_allclose(dq_fd, dq, rtol=RTOL, atol=ATOL)


    def test_dl_dP(self, verbose=False):
        n, m = 10, 10
        for diff_mode in diff_modes:

            # Get gradients
            [P, q, A, l, u, true_x], [dP, dq, dA, dl, du] = self.get_grads(
                n=n, m=m, P_scale=100., A_scale=100., diff_mode=diff_mode)

            P_idx = P.nonzero()

            def f(P_val):
                P_qp = sparse.csc_matrix((P_val, P_idx), shape=P.shape)
                # Go from values to sparse matrix
                m = osqp.OSQP()
                m.setup(P_qp, q, A, l, u, eps_abs=eps_abs, eps_rel=eps_rel, verbose=False)
                res = m.solve()
                x_hat = res.x

                return 0.5 * np.sum(np.square(x_hat - true_x))

            dP_fd_val = nd.Gradient(f)(P.data)  # Take numerical gradient of f
            dP_fd = sparse.csc_matrix((dP_fd_val, P_idx), shape=P.shape)
            dP_fd = (dP_fd + dP_fd.T)/2
            if verbose:
                print('--- ' + diff_mode)
                print('dP_fd: ', np.round(dP_fd.data, decimals=4))
                print('dP: ', np.round(dP.data, decimals=4))

            npt.assert_allclose(dP.todense(), dP_fd.todense(), rtol=RTOL, atol=ATOL)



    def test_dl_dA(self, verbose=True):
        n, m = 3, 3
        for diff_mode in diff_modes:

            # Get gradients
            [P, q, A, l, u, true_x], [dP, dq, dA, dl, du] = self.get_grads(
                n=n, m=m, P_scale=100., A_scale=100., diff_mode=diff_mode)

            A_idx = A.nonzero()

            def f(A_val):
                A_qp = sparse.csc_matrix((A_val, A_idx), shape=A.shape)
                # Go from values to sparse matrix
                m = osqp.OSQP()
                m.setup(P, q, A_qp, l, u, eps_abs=eps_abs, eps_rel=eps_rel, verbose=False)
                res = m.solve()
                x_hat = res.x

                return 0.5 * np.sum(np.square(x_hat - true_x))

            dA_fd_val = nd.Gradient(f)(A.data)  # Take numerical gradient of f
            dA_fd = sparse.csc_matrix((dA_fd_val, A_idx), shape=A.shape)
            if verbose:
                print('--- ' + diff_mode)
                print('dA_fd: ', np.round(dA_fd.data, decimals=4))
                print('dA: ', np.round(dA.data, decimals=4))

            npt.assert_allclose(dA.todense(), dA_fd.todense(), rtol=RTOL, atol=ATOL)

    def test_dl_dl(self, verbose=True):
        n, m = 10, 10
        for diff_mode in diff_modes:

            # Get gradients
            [P, q, A, l, u, true_x], [dP, dq, dA, dl, du] = self.get_grads(
                n=n, m=m, P_scale=100., A_scale=100., diff_mode=diff_mode)

            def f(l):
                m = osqp.OSQP()
                m.setup(P, q, A, l, u, eps_abs=eps_abs, eps_rel=eps_rel, verbose=False)
                res = m.solve()
                x_hat = res.x

                return 0.5 * np.sum(np.square(x_hat - true_x))

            dl_fd = nd.Gradient(f)(l)  # Take numerical gradient of f
            if verbose:
                print('--- ' + diff_mode)
                print('dl_fd: ', np.round(dl_fd, decimals=4))
                print('dl: ', np.round(dl, decimals=4))
            npt.assert_allclose(dl_fd, dl, rtol=RTOL, atol=ATOL)

    def test_dl_du(self, verbose=True):
        n, m = 5, 5
        for diff_mode in diff_modes:

            # Get gradients
            [P, q, A, l, u, true_x], [dP, dq, dA, dl, du] = self.get_grads(
                n=n, m=m, P_scale=100., A_scale=100., diff_mode=diff_mode)

            def f(u):
                m = osqp.OSQP()
                m.setup(P, q, A, l, u, eps_abs=eps_abs, eps_rel=eps_rel, verbose=False)
                res = m.solve()
                x_hat = res.x

                return 0.5 * np.sum(np.square(x_hat - true_x))

            du_fd = nd.Gradient(f)(u)  # Take numerical gradient of f
            if verbose:
                print('--- ' + diff_mode)
                print('du_fd: ', np.round(du_fd, decimals=4))
                print('du: ', np.round(du, decimals=4))
            npt.assert_allclose(du_fd, du, rtol=RTOL, atol=ATOL)
