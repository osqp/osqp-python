# Test osqp python module
from collections import deque
import osqp
# import osqppurepy as osqp
import numpy as np
import numpy.random as npr
from scipy import sparse
from scipy.optimize import approx_fprime
import numpy.testing as npt

# Unit Test
import unittest
import pdb


npr.seed(1)

# Tests settings
grad_precision = 1e-5
rel_tol = 1e-3
abs_tol = 1e-3
# rel_tol = 1e-3
# abs_tol = 1e-3

# OSQP settings
eps_abs = 1e-8
eps_rel = 1e-8
max_iter = 50000


class derivative_tests(unittest.TestCase):

    def get_prob(self, n=10, m=3, P_scale=1., A_scale=1.):
        L = np.random.randn(n, n-1)
        # P = sparse.csc_matrix(L.dot(L.T) + 5. * sparse.eye(n))
        # P = sparse.csc_matrix(L.dot(L.T) + 0.01 * sparse.eye(n))
        P = sparse.csc_matrix(L.dot(L.T))
        x_0 = npr.randn(n)
        s_0 = npr.rand(m)
        A = sparse.csc_matrix(npr.randn(m, n))
        u = A.dot(x_0) + s_0
        # l = -10 - 10 * npr.rand(m)
        l = A.dot(x_0) - s_0
        q = npr.randn(n)
        true_x = npr.randn(n)

        return [P, q,  A, l, u, true_x]

    def get_grads(self, P, q, A, l, u, true_x, mode='qdldl'):
        # Get gradients by solving with osqp
        m = osqp.OSQP(eps_rel=1e-8, eps_abs=1e-8)
        m.setup(P, q, A, l, u, eps_abs=eps_abs, eps_rel=eps_rel,
                max_iter=max_iter, verbose=False)
        results = m.solve()
        if results.info.status != "solved":
            raise ValueError("Problem not solved!")
        x = results.x
        grads = m.adjoint_derivative(dx=x - true_x, mode=mode)
        # grads = m.adjoint_derivative(dx=np.ones(x.size))

        return grads

    def get_forward_grads(self, P, q, A, l, u, dP, dq, dA, dl, du, mode='qdldl'):
        # Get gradients by solving with osqp
        m = osqp.OSQP(eps_rel=1e-8, eps_abs=1e-8)
        m.setup(P, q, A, l, u, eps_abs=eps_abs, eps_rel=eps_rel,
                max_iter=max_iter, verbose=False)
        results = m.solve()
        if results.info.status != "solved":
            raise ValueError("Problem not solved!")
        grads = m.forward_derivative(
            dP=dP, dq=dq, dA=dA, dl=dl, du=du, mode=mode)
        return grads

    def test_dsol_dq(self, verbose=False):
        n, m = 5, 5

        prob = self.get_prob(n=n, m=m, P_scale=100., A_scale=100.)
        P, q, A, l, u, true_x = prob

        def grad(dq, mode):
            [dx, dyl, dyu] = self.get_forward_grads(
                P, q, A, l, u, None, dq, None, None, None, mode=mode)
            return dx, dyl, dyu

        dq = np.random.normal(size=(n))
        dx_qdldl, dyl_qdldl, dyu_qdldl = grad(dq, 'qdldl')
        dx_lsqr, dyl_lsqr, dyu_lsqr = grad(dq, 'lsqr')
        osqp_solver = osqp.OSQP()
        osqp_solver.setup(P, q, A, l, u, eps_abs=eps_abs, eps_rel=eps_rel,
                          max_iter=max_iter, verbose=False)
        res = osqp_solver.solve()
        if res.info.status != "solved":
            raise ValueError("Problem not solved!")
        x1 = res.x
        y1 = res.y

        eps = grad_precision
        osqp_solver.setup(P, q + eps*dq, A, l, u, eps_abs=eps_abs, eps_rel=eps_rel,
                          max_iter=max_iter, verbose=False)
        res = osqp_solver.solve()
        if res.info.status != "solved":
            raise ValueError("Problem not solved!")
        x2 = res.x
        y2 = res.y

        dx_fd = (x2-x1)/eps
        dy_fd = (y2-y1)/eps
        dyl_fd = np.zeros(m)
        dyl_fd[y1 < 0] = -dy_fd[y1 < 0]
        dyu_fd = np.zeros(m)
        dyu_fd[y1 >= 0] = dy_fd[y1 >= 0]

        if verbose:
            print('dx_fd: ', np.round(dx_fd, decimals=4))
            print('dx: ', np.round(dx_qdldl, decimals=4))

        npt.assert_allclose(dx_fd, dx_qdldl, rtol=rel_tol, atol=abs_tol)
        npt.assert_allclose(dyl_fd, dyl_qdldl, rtol=rel_tol, atol=abs_tol)
        npt.assert_allclose(dyu_fd, dyu_qdldl, rtol=rel_tol, atol=abs_tol)

        npt.assert_allclose(dx_fd, dx_lsqr, rtol=rel_tol, atol=abs_tol)
        npt.assert_allclose(dyl_fd, dyl_lsqr, rtol=rel_tol, atol=abs_tol)
        npt.assert_allclose(dyu_fd, dyu_lsqr, rtol=rel_tol, atol=abs_tol)

    def test_eq_inf_forward(self, verbose=False):
        n, m = 10, 10

        prob = self.get_prob(n=n, m=m, P_scale=100., A_scale=100.)
        P, q, A, l, u, true_x = prob
        l[:5] = u[:5]
        l[5:] = -osqp.constant('OSQP_INFTY')

        def grad(dq, mode):
            [dx, dyl, dyu] = self.get_forward_grads(
                P, q, A, l, u, None, dq, None, None, None, mode=mode)
            return dx, dyl, dyu

        dq = np.random.normal(size=(n))
        dx_qdldl, dyl_qdldl, dyu_qdldl = grad(dq, 'qdldl')
        dx_lsqr, dyl_lsqr, dyu_lsqr = grad(dq, 'lsqr')
        osqp_solver = osqp.OSQP()
        osqp_solver.setup(P, q, A, l, u, eps_abs=eps_abs, eps_rel=eps_rel,
                          max_iter=max_iter, verbose=False)
        res = osqp_solver.solve()
        if res.info.status != "solved":
            raise ValueError("Problem not solved!")
        x1 = res.x
        y1 = res.y

        eps = grad_precision
        osqp_solver.setup(P, q + eps*dq, A, l, u, eps_abs=eps_abs, eps_rel=eps_rel,
                          max_iter=max_iter, verbose=False)
        res = osqp_solver.solve()
        if res.info.status != "solved":
            raise ValueError("Problem not solved!")
        x2 = res.x
        y2 = res.y

        dx_fd = (x2-x1)/eps
        dy_fd = (y2-y1)/eps
        dyl_fd = np.zeros(m)
        dyl_fd[y1 < 0] = -dy_fd[y1 < 0]
        dyu_fd = np.zeros(m)
        dyu_fd[y1 >= 0] = dy_fd[y1 >= 0]

        if verbose:
            print('dx_fd: ', np.round(dx_fd, decimals=4))
            print('dx: ', np.round(dx_qdldl, decimals=4))

        npt.assert_allclose(dx_fd, dx_qdldl, rtol=rel_tol, atol=abs_tol)
        npt.assert_allclose(dyl_fd, dyl_qdldl, rtol=rel_tol, atol=abs_tol)
        npt.assert_allclose(dyu_fd, dyu_qdldl, rtol=rel_tol, atol=abs_tol)

        npt.assert_allclose(dx_fd, dx_lsqr, rtol=rel_tol, atol=abs_tol)
        npt.assert_allclose(dyl_fd, dyl_lsqr, rtol=rel_tol, atol=abs_tol)
        npt.assert_allclose(dyu_fd, dyu_lsqr, rtol=rel_tol, atol=abs_tol)


    def test_multiple_forward_derivative(self, verbose=False):
        n, m = 5, 5

        prob = self.get_prob(n=n, m=m, P_scale=100., A_scale=100.)
        P, q, A, l, u, true_x = prob

        def grad(dP, dq, dA, dl, du, mode):
            [dx, dyl, dyu] = self.get_forward_grads(
                P, q, A, l, u, dP, dq, dA, dl, du, mode=mode)
            return dx, dyl, dyu

        dq = np.random.normal(size=(n))
        dA = sparse.csc_matrix(np.random.normal(size=(m, n)))
        dl = np.random.normal(size=(m))
        du = np.random.normal(size=(m))
        dL = np.random.normal(size=(n, n))
        dP = dL + dL.T
        dx_qdldl, dyl_qdldl, dyu_qdldl = grad(dP, dq, dA, dl, du, 'qdldl')
        dx_lsqr, dyl_lsqr, dyu_lsqr = grad(dP, dq, dA, dl, du, 'lsqr')
        osqp_solver = osqp.OSQP()
        osqp_solver.setup(P, q, A, l, u, eps_abs=eps_abs, eps_rel=eps_rel,
                          max_iter=max_iter, verbose=False)
        res = osqp_solver.solve()
        if res.info.status != "solved":
            raise ValueError("Problem not solved!")
        x1 = res.x
        y1 = res.y

        eps = grad_precision
        osqp_solver.setup(P + eps*dP, q + eps*dq, A + eps*dA, l + eps*dl, u + eps*du, eps_abs=eps_abs, eps_rel=eps_rel,
                          max_iter=max_iter, verbose=False)
        res = osqp_solver.solve()
        if res.info.status != "solved":
            raise ValueError("Problem not solved!")
        x2 = res.x
        y2 = res.y

        dx_fd = (x2-x1)/eps
        dy_fd = (y2-y1)/eps
        dyl_fd = np.zeros(m)
        dyl_fd[y1 < 0] = -dy_fd[y1 < 0]
        dyu_fd = np.zeros(m)
        dyu_fd[y1 >= 0] = dy_fd[y1 >= 0]

        if verbose:
            print('dx_fd: ', np.round(dx_fd, decimals=4))
            print('dx: ', np.round(dx_qdldl, decimals=4))

        npt.assert_allclose(dx_fd, dx_qdldl, rtol=rel_tol, atol=abs_tol)
        npt.assert_allclose(dyl_fd, dyl_qdldl, rtol=rel_tol, atol=abs_tol)
        npt.assert_allclose(dyu_fd, dyu_qdldl, rtol=rel_tol, atol=abs_tol)

        npt.assert_allclose(dx_fd, dx_lsqr, rtol=rel_tol, atol=abs_tol)
        npt.assert_allclose(dyl_fd, dyl_lsqr, rtol=rel_tol, atol=abs_tol)
        npt.assert_allclose(dyu_fd, dyu_lsqr, rtol=rel_tol, atol=abs_tol)

    def test_dl_dq(self, verbose=False):
        n, m = 5, 5

        prob = self.get_prob(n=n, m=m, P_scale=100., A_scale=100.)
        P, q, A, l, u, true_x = prob

        def grad(q):
            [dP, dq, dA, dl, du] = self.get_grads(P, q, A, l, u, true_x)
            return dq

        def f(q):
            m = osqp.OSQP()
            m.setup(P, q, A, l, u, eps_abs=eps_abs, eps_rel=eps_rel,
                    max_iter=max_iter, verbose=False)
            res = m.solve()
            if res.info.status != "solved":
                raise ValueError("Problem not solved!")
            x_hat = res.x

            return 0.5 * np.sum(np.square(x_hat - true_x))

        dq = grad(q)
        dq_fd = approx_fprime(q, f, grad_precision)

        if verbose:
            print('dq_fd: ', np.round(dq_fd, decimals=4))
            print('dq: ', np.round(dq, decimals=4))

        npt.assert_allclose(dq_fd, dq, rtol=rel_tol, atol=abs_tol)

    def test_dl_dq(self, verbose=False):
        n, m = 5, 5

        prob = self.get_prob(n=n, m=m, P_scale=100., A_scale=100.)
        P, q, A, l, u, true_x = prob

        def grad(q):
            [dP, dq, dA, dl, du] = self.get_grads(P, q, A, l, u, true_x)
            return dq

        def f(q):
            m = osqp.OSQP()
            m.setup(P, q, A, l, u, eps_abs=eps_abs, eps_rel=eps_rel,
                    max_iter=max_iter, verbose=False)
            res = m.solve()
            if res.info.status != "solved":
                raise ValueError("Problem not solved!")
            x_hat = res.x

            return 0.5 * np.sum(np.square(x_hat - true_x))

        dq = grad(q)
        dq_fd = approx_fprime(q, f, grad_precision)

        if verbose:
            print('dq_fd: ', np.round(dq_fd, decimals=4))
            print('dq: ', np.round(dq, decimals=4))

        npt.assert_allclose(dq_fd, dq, rtol=rel_tol, atol=abs_tol)

    def test_dl_dP(self, verbose=False):
        n, m = 3, 3

        prob = self.get_prob(n=n, m=m, P_scale=100., A_scale=100.)
        P, q, A, l, u, true_x = prob
        P_idx = P.nonzero()

        def grad(P_val):
            P_qp = sparse.csc_matrix((P_val, P_idx), shape=P.shape)
            [dP, dq, dA, dl, du] = self.get_grads(P_qp, q, A, l, u, true_x)
            return dP

        def f(P_val):
            P_qp = sparse.csc_matrix((P_val, P_idx), shape=P.shape)
            m = osqp.OSQP()
            m.setup(P_qp, q, A, l, u, eps_abs=eps_abs,
                    eps_rel=eps_rel, max_iter=max_iter, verbose=False)
            res = m.solve()
            if res.info.status != "solved":
                raise ValueError("Problem not solved!")
            x_hat = res.x

            return 0.5 * np.sum(np.square(x_hat - true_x))

        dP = grad(P.data)
        dP_fd_val = approx_fprime(P.data, f, grad_precision)
        dP_fd = sparse.csc_matrix((dP_fd_val, P_idx), shape=P.shape)
        dP_fd = (dP_fd + dP_fd.T)/2

        if verbose:
            print('dP_fd: ', np.round(dP_fd.data, decimals=4))
            print('dA: ', np.round(dP.data, decimals=4))

        npt.assert_allclose(dP.todense(), dP_fd.todense(),
                            rtol=rel_tol, atol=abs_tol)

    def test_dl_dA(self, verbose=False):
        n, m = 3, 3

        prob = self.get_prob(n=n, m=m, P_scale=100., A_scale=100.)
        P, q, A, l, u, true_x = prob
        A_idx = A.nonzero()

        def grad(A_val):
            A_qp = sparse.csc_matrix((A_val, A_idx), shape=A.shape)
            [dP, dq, dA, dl, du] = self.get_grads(P, q, A_qp, l, u, true_x)
            return dA

        def f(A_val):
            A_qp = sparse.csc_matrix((A_val, A_idx), shape=A.shape)
            m = osqp.OSQP()
            m.setup(P, q, A_qp, l, u, eps_abs=eps_abs,
                    eps_rel=eps_rel, max_iter=max_iter, verbose=False)
            res = m.solve()
            if res.info.status != "solved":
                raise ValueError("Problem not solved!")
            x_hat = res.x

            return 0.5 * np.sum(np.square(x_hat - true_x))

        dA = grad(A.data)
        dA_fd_val = approx_fprime(A.data, f, grad_precision)
        dA_fd = sparse.csc_matrix((dA_fd_val, A_idx), shape=A.shape)

        if verbose:
            print('dA_fd: ', np.round(dA_fd.data, decimals=4))
            print('dA: ', np.round(dA.data, decimals=4))

        npt.assert_allclose(dA.todense(), dA_fd.todense(),
                            rtol=rel_tol, atol=abs_tol)

    def test_dl_dl(self, verbose=False):
        n, m = 30, 30

        prob = self.get_prob(n=n, m=m, P_scale=100., A_scale=100.)
        P, q, A, l, u, true_x = prob

        def grad(l, mode):
            [dP, dq, dA, dl, du] = self.get_grads(
                P, q, A, l, u, true_x, mode=mode)
            return dl

        def f(l):
            m = osqp.OSQP()
            m.setup(P, q, A, l, u, eps_abs=eps_abs, eps_rel=eps_rel,
                    max_iter=max_iter, verbose=False)
            res = m.solve()
            if res.info.status != "solved":
                raise ValueError("Problem not solved!")
            x_hat = res.x

            return 0.5 * np.sum(np.square(x_hat - true_x))

        dl_lsqr = grad(l, 'lsqr')
        dl_qdldl = grad(l, 'qdldl')
        dl_fd = approx_fprime(l, f, grad_precision)

        if verbose:
            print('dl_fd: ', np.round(dl_fd, decimals=4).tolist())
            print('dl_lsqr: ', np.round(dl_lsqr, decimals=4).tolist())
            print('dl_qdldl: ', np.round(dl_qdldl, decimals=4).tolist())

        npt.assert_allclose(dl_fd, dl_lsqr,
                            rtol=rel_tol, atol=abs_tol)
        npt.assert_allclose(dl_fd, dl_qdldl,
                            rtol=rel_tol, atol=abs_tol)

    def test_dl_du(self, verbose=False):
        n, m = 10, 20

        prob = self.get_prob(n=n, m=m, P_scale=100., A_scale=100.)
        P, q, A, l, u, true_x = prob

        def grad(u, mode):
            [dP, dq, dA, dl, du] = self.get_grads(
                P, q, A, l, u, true_x, mode=mode)
            return du

        def f(u):
            m = osqp.OSQP()
            m.setup(P, q, A, l, u, eps_abs=eps_abs, eps_rel=eps_rel,
                    max_iter=max_iter, verbose=False)
            res = m.solve()
            if res.info.status != "solved":
                raise ValueError("Problem not solved!")
            x_hat = res.x

            return 0.5 * np.sum(np.square(x_hat - true_x))

        du_lsqr = grad(u, 'lsqr')
        du_fd = approx_fprime(u, f, grad_precision)

        if verbose:
            print('du_fd: ', np.round(du_fd, decimals=4))
            print('du: ', np.round(du_lsqr, decimals=4))

        npt.assert_allclose(du_fd, du_lsqr,
                            rtol=rel_tol, atol=abs_tol)

    def test_dl_dA_eq(self, verbose=False):
        n, m = 40, 40

        prob = self.get_prob(n=n, m=m, P_scale=100., A_scale=100.)
        P, q, A, l, u, true_x = prob
        # u = l
        # l[0:10] = -osqp.constant('OSQP_INFTY')
        u[:20] = l[:20]

        A_idx = A.nonzero()

        def grad(A_val, mode):
            A_qp = sparse.csc_matrix((A_val, A_idx), shape=A.shape)
            [dP, dq, dA, dl, du] = self.get_grads(
                P, q, A_qp, l, u, true_x, mode=mode)
            return dA

        def f(A_val):
            A_qp = sparse.csc_matrix((A_val, A_idx), shape=A.shape)
            m = osqp.OSQP()
            m.setup(P, q, A_qp, l, u, eps_abs=eps_abs,
                    eps_rel=eps_rel, max_iter=max_iter, verbose=False)
            res = m.solve()
            if res.info.status != "solved":
                raise ValueError("Problem not solved!")
            x_hat = res.x

            return 0.5 * np.sum(np.square(x_hat - true_x))
            # return np.sum(x_hat)

        dA_lsqr = grad(A.data, 'lsqr')
        dA_qdldl = grad(A.data, 'qdldl')
        dA_fd_val = approx_fprime(A.data, f, grad_precision)
        dA_fd = sparse.csc_matrix((dA_fd_val, A_idx), shape=A.shape)

        if verbose:
            print('dA_fd: ', np.round(dA_fd.data, decimals=6))
            print('dA_lsqr: ', np.round(dA_lsqr.data, decimals=6))
            print('dA_qdldl: ', np.round(dA_qdldl.data, decimals=6))

        npt.assert_allclose(dA_lsqr.todense(), dA_fd.todense(),
                            rtol=rel_tol, atol=abs_tol)
        npt.assert_allclose(dA_qdldl.todense(), dA_fd.todense(),
                            rtol=rel_tol, atol=abs_tol)

    def test_dl_dq_eq(self, verbose=False):
        n, m = 20, 15

        prob = self.get_prob(n=n, m=m, P_scale=1., A_scale=1.)
        P, q, A, l, u, true_x = prob
        # u = l
        # l[20:40] = -osqp.constant('OSQP_INFTY')
        u[:20] = l[:20]

        A_idx = A.nonzero()

        def grad(q, mode):
            [dP, dq, dA, dl, du] = self.get_grads(
                P, q, A, l, u, true_x, mode=mode)
            return dq

        def f(q):
            m = osqp.OSQP()
            m.setup(P, q, A, l, u, eps_abs=eps_abs, eps_rel=eps_rel,
                    max_iter=max_iter, verbose=False)
            res = m.solve()
            if res.info.status != "solved":
                raise ValueError("Problem not solved!")
            x_hat = res.x

            return 0.5 * np.sum(np.square(x_hat - true_x))

        dq_lsqr = grad(q, 'lsqr')
        dq_qdldl = grad(q, 'qdldl')
        dq_fd = approx_fprime(q, f, grad_precision)

        if verbose:
            print('dq_fd: ', np.round(dq_fd, decimals=4))
            print('dq_qdldl: ', np.round(dq_qdldl, decimals=4))
            print('dq_lsqr: ', np.round(dq_lsqr, decimals=4))

        npt.assert_allclose(dq_fd, dq_lsqr, rtol=rel_tol, atol=abs_tol)
        npt.assert_allclose(dq_fd, dq_qdldl, rtol=rel_tol, atol=abs_tol)

    def test_dl_dq_eq_large(self, verbose=False):
        n, m = 100, 120

        prob = self.get_prob(n=n, m=m, P_scale=1., A_scale=1.)
        P, q, A, l, u, true_x = prob

        l[20:40] = -osqp.constant('OSQP_INFTY')
        u[:20] = l[:20]

        A_idx = A.nonzero()

        def grad(q, mode):
            [dP, dq, dA, dl, du] = self.get_grads(
                P, q, A, l, u, true_x, mode=mode)
            return dq

        def f(q):
            m = osqp.OSQP()
            m.setup(P, q, A, l, u, eps_abs=eps_abs, eps_rel=eps_rel,
                    max_iter=max_iter, verbose=False)
            res = m.solve()
            if res.info.status != "solved":
                raise ValueError("Problem not solved!")
            x_hat = res.x

            return 0.5 * np.sum(np.square(x_hat - true_x))

        dq_lsqr = grad(q, 'lsqr')
        dq_qdldl = grad(q, 'qdldl')
        dq_fd = approx_fprime(q, f, grad_precision)

        if verbose:
            print('dq_fd: ', np.round(dq_fd, decimals=4))
            print('dq_qdldl: ', np.round(dq_qdldl, decimals=4))
            print('dq_lsqr: ', np.round(dq_lsqr, decimals=4))

        npt.assert_allclose(dq_fd, dq_lsqr, rtol=rel_tol, atol=abs_tol)
        npt.assert_allclose(dq_fd, dq_qdldl, rtol=rel_tol, atol=abs_tol)
