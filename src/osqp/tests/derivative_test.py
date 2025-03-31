import osqp
import numpy as np
import numpy.random as npr
from scipy import sparse
from scipy.optimize import approx_fprime
import numpy.testing as npt
import unittest
import pytest


npr.seed(1)

# Tests settings
grad_precision = 1e-5
rel_tol = 5e-3
abs_tol = 5e-3
rel_tol_relaxed = 1e-2
abs_tol_relaxed = 1e-2

# OSQP settings
eps_abs = 1e-9
eps_rel = 1e-9
max_iter = 500000


@pytest.mark.skipif(not osqp.algebra_available('builtin'), reason='Builtin Algebra not available')
class derivative_tests(unittest.TestCase):
    def setUp(self):
        npr.seed(1)

    def get_prob(self, n=10, m=3, P_scale=1.0, A_scale=1.0):
        L = np.random.randn(n, n - 1)
        # P = sparse.csc_matrix(L.dot(L.T) + 5. * sparse.eye(n))
        P = sparse.csc_matrix(L.dot(L.T) + 0.1 * sparse.eye(n))
        # P = sparse.csc_matrix(L.dot(L.T))
        x_0 = npr.randn(n)
        s_0 = npr.rand(m)
        A = sparse.csc_matrix(npr.randn(m, n))
        u = A.dot(x_0) + s_0
        # l = -10 - 10 * npr.rand(m)
        l = A.dot(x_0) - s_0
        q = npr.randn(n)
        true_x = npr.randn(n)
        true_y = npr.randn(m)

        return [P, q, A, l, u, true_x, true_y]

    def get_grads(self, P, q, A, l, u, true_x, true_y=None):
        # Get gradients by solving with osqp
        m = osqp.OSQP(algebra='builtin')
        m.setup(
            P,
            q,
            A,
            l,
            u,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            max_iter=max_iter,
            verbose=True,
        )
        results = m.solve()
        if results.info.status_val != osqp.SolverStatus.OSQP_SOLVED:
            raise ValueError('Problem not solved!')
        x = results.x
        y = results.y
        if true_y is None:
            m.adjoint_derivative_compute(dx=x - true_x)
        else:
            m.adjoint_derivative_compute(dx=x - true_x, dy=y - true_y)

        dP, dA = m.adjoint_derivative_get_mat()
        dq, dl, du = m.adjoint_derivative_get_vec()
        grads = dP, dq, dA, dl, du

        return grads

    def get_forward_grads(self, P, q, A, l, u, dP, dq, dA, dl, du):
        # Get gradients by solving with osqp
        m = osqp.OSQP(algebra='builtin', eps_rel=eps_rel, eps_abs=eps_abs)
        m.setup(
            P,
            q,
            A,
            l,
            u,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            max_iter=max_iter,
            verbose=False,
        )
        results = m.solve()
        if results.info.status_val != osqp.SolverStatus.OSQP_SOLVED:
            raise ValueError('Problem not solved!')
        grads = m.forward_derivative(dP=dP, dq=dq, dA=dA, dl=dl, du=du)
        return grads

    @pytest.mark.skip(reason='forward derivatives not implemented yet')
    def test_dsol_dq(self, verbose=False):
        n, m = 5, 5

        prob = self.get_prob(n=n, m=m, P_scale=100.0, A_scale=100.0)
        P, q, A, l, u, true_x, true_y = prob

        def grad(dq):
            [dx, dyl, dyu] = self.get_forward_grads(P, q, A, l, u, None, dq, None, None, None)
            return dx, dyl, dyu

        dq = np.random.normal(size=(n))
        dx_computed, dyl_computed, dyu_computed = grad(dq)

        osqp_solver = osqp.OSQP(algebra='builtin')
        osqp_solver.setup(
            P,
            q,
            A,
            l,
            u,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            max_iter=max_iter,
            verbose=False,
        )
        res = osqp_solver.solve()
        if res.info.status_val != osqp.SolverStatus.OSQP_SOLVED:
            raise ValueError('Problem not solved!')
        x1 = res.x
        y1 = res.y

        eps = grad_precision
        osqp_solver.setup(
            P,
            q + eps * dq,
            A,
            l,
            u,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            max_iter=max_iter,
            verbose=False,
        )
        res = osqp_solver.solve()
        if res.info.status_val != osqp.SolverStatus.OSQP_SOLVED:
            raise ValueError('Problem not solved!')
        x2 = res.x
        y2 = res.y

        dx_fd = (x2 - x1) / eps
        dy_fd = (y2 - y1) / eps
        dyl_fd = np.zeros(m)
        dyl_fd[y1 < 0] = -dy_fd[y1 < 0]
        dyu_fd = np.zeros(m)
        dyu_fd[y1 >= 0] = dy_fd[y1 >= 0]

        if verbose:
            print('dx_fd: ', np.round(dx_fd, decimals=4))
            print('dx: ', np.round(dx_computed, decimals=4))

        npt.assert_allclose(dx_fd, dx_computed, rtol=rel_tol, atol=abs_tol)
        npt.assert_allclose(dyl_fd, dyl_computed, rtol=rel_tol, atol=abs_tol)
        npt.assert_allclose(dyu_fd, dyu_computed, rtol=rel_tol, atol=abs_tol)

    @pytest.mark.skip(reason='forward derivatives not implemented yet')
    def test_eq_inf_forward(self, verbose=False):
        n, m = 10, 10

        prob = self.get_prob(n=n, m=m, P_scale=100.0, A_scale=100.0)
        P, q, A, l, u, true_x, true_y = prob
        l[:5] = u[:5]
        l[5:] = -osqp.constant('OSQP_INFTY', algebra='builtin')

        def grad(dq):
            [dx, dyl, dyu] = self.get_forward_grads(P, q, A, l, u, None, dq, None, None, None)
            return dx, dyl, dyu

        dq = np.random.normal(size=(n))
        dx_computed, dyl_computed, dyu_computed = grad(dq)
        osqp_solver = osqp.OSQP(algebra='builtin')
        osqp_solver.setup(
            P,
            q,
            A,
            l,
            u,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            max_iter=max_iter,
            verbose=False,
        )
        res = osqp_solver.solve()
        if res.info.status_val != osqp.SolverStatus.OSQP_SOLVED:
            raise ValueError('Problem not solved!')
        x1 = res.x
        y1 = res.y

        eps = grad_precision
        osqp_solver.setup(
            P,
            q + eps * dq,
            A,
            l,
            u,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            max_iter=max_iter,
            verbose=False,
        )
        res = osqp_solver.solve()
        if res.info.status_val != osqp.SolverStatus.OSQP_SOLVED:
            raise ValueError('Problem not solved!')
        x2 = res.x
        y2 = res.y

        dx_fd = (x2 - x1) / eps
        dy_fd = (y2 - y1) / eps
        dyl_fd = np.zeros(m)
        dyl_fd[y1 < 0] = -dy_fd[y1 < 0]
        dyu_fd = np.zeros(m)
        dyu_fd[y1 >= 0] = dy_fd[y1 >= 0]

        if verbose:
            print('dx_fd: ', np.round(dx_fd, decimals=4))
            print('dx: ', np.round(dx_computed, decimals=4))

        npt.assert_allclose(dx_fd, dx_computed, rtol=rel_tol, atol=abs_tol)
        npt.assert_allclose(dyl_fd, dyl_computed, rtol=rel_tol, atol=abs_tol)
        npt.assert_allclose(dyu_fd, dyu_computed, rtol=rel_tol, atol=abs_tol)

    @pytest.mark.skip(reason='forward derivatives not implemented yet')
    def test_multiple_forward_derivative(self, verbose=False):
        n, m = 5, 5

        prob = self.get_prob(n=n, m=m, P_scale=100.0, A_scale=100.0)
        P, q, A, l, u, true_x, true_y = prob

        def grad(dP, dq, dA, dl, du):
            [dx, dyl, dyu] = self.get_forward_grads(P, q, A, l, u, dP, dq, dA, dl, du)
            return dx, dyl, dyu

        dq = np.random.normal(size=(n))
        dA = sparse.csc_matrix(np.random.normal(size=(m, n)))
        dl = np.random.normal(size=(m))
        du = np.random.normal(size=(m))
        dL = np.random.normal(size=(n, n))
        dP = dL + dL.T
        dx_computed, dyl_computed, dyu_computed = grad(dP, dq, dA, dl, du)
        osqp_solver = osqp.OSQP(algebra='builtin')
        osqp_solver.setup(
            P,
            q,
            A,
            l,
            u,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            max_iter=max_iter,
            verbose=False,
        )
        res = osqp_solver.solve()
        if res.info.status_val != osqp.SolverStatus.OSQP_SOLVED:
            raise ValueError('Problem not solved!')
        x1 = res.x
        y1 = res.y

        eps = grad_precision
        osqp_solver.setup(
            P + eps * dP,
            q + eps * dq,
            A + eps * dA,
            l + eps * dl,
            u + eps * du,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            max_iter=max_iter,
            verbose=False,
        )
        res = osqp_solver.solve()
        if res.info.status_val != osqp.SolverStatus.OSQP_SOLVED:
            raise ValueError('Problem not solved!')
        x2 = res.x
        y2 = res.y

        dx_fd = (x2 - x1) / eps
        dy_fd = (y2 - y1) / eps
        dyl_fd = np.zeros(m)
        dyl_fd[y1 < 0] = -dy_fd[y1 < 0]
        dyu_fd = np.zeros(m)
        dyu_fd[y1 >= 0] = dy_fd[y1 >= 0]

        if verbose:
            print('dx_fd: ', np.round(dx_fd, decimals=4))
            print('dx: ', np.round(dx_computed, decimals=4))

        npt.assert_allclose(dx_fd, dx_computed, rtol=rel_tol, atol=abs_tol)
        npt.assert_allclose(dyl_fd, dyl_computed, rtol=rel_tol, atol=abs_tol)
        npt.assert_allclose(dyu_fd, dyu_computed, rtol=rel_tol, atol=abs_tol)

    def test_dl_dq(self, verbose=False):
        n, m = 5, 5

        prob = self.get_prob(n=n, m=m, P_scale=100.0, A_scale=100.0)
        P, q, A, l, u, true_x, true_y = prob

        def grad(q):
            dP, dq, _, _, _ = self.get_grads(P, q, A, l, u, true_x)
            return dq

        def f(q):
            m = osqp.OSQP(algebra='builtin')
            m.setup(
                P,
                q,
                A,
                l,
                u,
                eps_abs=eps_abs,
                eps_rel=eps_rel,
                max_iter=max_iter,
                verbose=False,
            )
            res = m.solve()
            if res.info.status_val != osqp.SolverStatus.OSQP_SOLVED:
                raise ValueError('Problem not solved!')
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

        prob = self.get_prob(n=n, m=m, P_scale=100.0, A_scale=100.0)
        P, q, A, l, u, true_x, true_y = prob
        P_idx = P.nonzero()

        def grad(P_val):
            P_qp = sparse.csc_matrix((P_val, P_idx), shape=P.shape)
            dP, _, _, _, _ = self.get_grads(P_qp, q, A, l, u, true_x)
            return dP

        def f(P_val):
            P_qp = sparse.csc_matrix((P_val, P_idx), shape=P.shape)
            m = osqp.OSQP(algebra='builtin')
            m.setup(
                P_qp,
                q,
                A,
                l,
                u,
                eps_abs=eps_abs,
                eps_rel=eps_rel,
                max_iter=max_iter,
                verbose=False,
            )
            res = m.solve()
            if res.info.status_val != osqp.SolverStatus.OSQP_SOLVED:
                raise ValueError('Problem not solved!')
            x_hat = res.x

            return 0.5 * np.sum(np.square(x_hat - true_x))

        dP = grad(P.data)
        dP_fd_val = approx_fprime(P.data, f, grad_precision)
        dP_fd = sparse.csc_matrix((dP_fd_val, P_idx), shape=P.shape)
        dP_fd = (dP_fd + dP_fd.T) / 2

        if verbose:
            print('dP_fd: ', np.round(dP_fd.data, decimals=4))
            print('dA: ', np.round(dP.data, decimals=4))

        npt.assert_allclose(np.triu(dP), np.triu(dP_fd.todense()), rtol=rel_tol, atol=abs_tol)

    def test_dl_dA(self, verbose=False):
        n, m = 3, 3

        prob = self.get_prob(n=n, m=m, P_scale=100.0, A_scale=100.0)
        P, q, A, l, u, true_x, true_y = prob
        A_idx = A.nonzero()

        def grad(A_val):
            A_qp = sparse.csc_matrix((A_val, A_idx), shape=A.shape)
            _, _, dA, _, _ = self.get_grads(P, q, A_qp, l, u, true_x)
            return dA

        def f(A_val):
            A_qp = sparse.csc_matrix((A_val, A_idx), shape=A.shape)
            m = osqp.OSQP(algebra='builtin')
            m.setup(
                P,
                q,
                A_qp,
                l,
                u,
                eps_abs=eps_abs,
                eps_rel=eps_rel,
                max_iter=max_iter,
                verbose=False,
            )
            res = m.solve()
            if res.info.status_val != osqp.SolverStatus.OSQP_SOLVED:
                raise ValueError('Problem not solved!')
            x_hat = res.x

            return 0.5 * np.sum(np.square(x_hat - true_x))

        dA = grad(A.data)
        dA_fd_val = approx_fprime(A.data, f, grad_precision)
        dA_fd = sparse.csc_matrix((dA_fd_val, A_idx), shape=A.shape)

        if verbose:
            print('dA_fd: ', np.round(dA_fd.data, decimals=4))
            print('dA: ', np.round(dA.data, decimals=4))

        npt.assert_allclose(dA, dA_fd.todense(), rtol=rel_tol, atol=abs_tol)

    def test_dl_dl(self, verbose=False):
        n, m = 30, 30

        prob = self.get_prob(n=n, m=m, P_scale=100.0, A_scale=100.0)
        P, q, A, l, u, true_x, true_y = prob

        def grad(l):
            _, _, _, dl, _ = self.get_grads(P, q, A, l, u, true_x)
            return dl

        def f(l):
            m = osqp.OSQP(algebra='builtin')
            m.setup(
                P,
                q,
                A,
                l,
                u,
                eps_abs=eps_abs,
                eps_rel=eps_rel,
                max_iter=max_iter,
                verbose=False,
            )
            res = m.solve()
            if res.info.status_val != osqp.SolverStatus.OSQP_SOLVED:
                raise ValueError('Problem not solved!')
            x_hat = res.x

            return 0.5 * np.sum(np.square(x_hat - true_x))

        dl_computed = grad(l)
        dl_fd = approx_fprime(l, f, grad_precision)

        if verbose:
            print('dl_fd: ', np.round(dl_fd, decimals=4).tolist())
            print('dl_computed: ', np.round(dl_computed, decimals=4).tolist())

        npt.assert_allclose(dl_fd, dl_computed, rtol=rel_tol, atol=abs_tol)

    def test_dl_du(self, verbose=False):
        n, m = 10, 20

        prob = self.get_prob(n=n, m=m, P_scale=100.0, A_scale=100.0)
        P, q, A, l, u, true_x, true_y = prob

        def grad(u):
            _, _, _, _, du = self.get_grads(P, q, A, l, u, true_x)
            return du

        def f(u):
            m = osqp.OSQP(algebra='builtin')
            m.setup(
                P,
                q,
                A,
                l,
                u,
                eps_abs=eps_abs,
                eps_rel=eps_rel,
                max_iter=max_iter,
                verbose=False,
            )
            res = m.solve()
            if res.info.status_val != osqp.SolverStatus.OSQP_SOLVED:
                raise ValueError('Problem not solved!')
            x_hat = res.x

            return 0.5 * np.sum(np.square(x_hat - true_x))

        du_computed = grad(u)
        du_fd = approx_fprime(u, f, grad_precision)

        if verbose:
            print('du_fd: ', np.round(du_fd, decimals=4))
            print('du: ', np.round(du_computed, decimals=4))

        npt.assert_allclose(du_fd, du_computed, rtol=rel_tol, atol=abs_tol)

    def test_dl_dA_eq(self, verbose=False):
        n, m = 30, 20

        prob = self.get_prob(n=n, m=m, P_scale=100.0, A_scale=100.0)
        P, q, A, l, u, true_x, true_y = prob
        # u = l
        # l[10:20] = -osqp.constant('OSQP_INFTY', algebra='builtin')
        u[:10] = l[:10]

        A_idx = A.nonzero()

        def grad(A_val):
            A_qp = sparse.csc_matrix((A_val, A_idx), shape=A.shape)
            _, _, dA, _, _ = self.get_grads(P, q, A_qp, l, u, true_x)
            return dA

        def f(A_val):
            A_qp = sparse.csc_matrix((A_val, A_idx), shape=A.shape)
            m = osqp.OSQP(algebra='builtin')
            m.setup(
                P,
                q,
                A_qp,
                l,
                u,
                eps_abs=eps_abs,
                eps_rel=eps_rel,
                max_iter=max_iter,
                verbose=False,
            )
            res = m.solve()
            if res.info.status_val != osqp.SolverStatus.OSQP_SOLVED:
                raise ValueError('Problem not solved!')
            x_hat = res.x

            return 0.5 * np.sum(np.square(x_hat - true_x))

        dA_computed = grad(A.data)
        dA_fd_val = approx_fprime(A.data, f, grad_precision)
        dA_fd = sparse.csc_matrix((dA_fd_val, A_idx), shape=A.shape)

        if verbose:
            print('dA_fd: ', np.round(dA_fd.data, decimals=6))
            print('dA_computed: ', np.round(dA_computed.data, decimals=6))

        npt.assert_allclose(dA_computed, dA_fd.todense(), rtol=rel_tol, atol=abs_tol)

    def test_dl_dq_eq(self, verbose=False):
        n, m = 20, 15

        prob = self.get_prob(n=n, m=m, P_scale=1.0, A_scale=1.0)
        P, q, A, l, u, true_x, true_y = prob
        # u = l
        # l[20:40] = -osqp.constant('OSQP_INFTY', algebra='builtin')
        u[:20] = l[:20]

        def grad(q):
            _, dq, _, _, _ = self.get_grads(P, q, A, l, u, true_x)
            return dq

        def f(q):
            m = osqp.OSQP(algebra='builtin')
            m.setup(
                P,
                q,
                A,
                l,
                u,
                eps_abs=eps_abs,
                eps_rel=eps_rel,
                max_iter=max_iter,
                verbose=False,
            )
            res = m.solve()
            if res.info.status_val != osqp.SolverStatus.OSQP_SOLVED:
                raise ValueError('Problem not solved!')
            x_hat = res.x

            return 0.5 * np.sum(np.square(x_hat - true_x))

        dq_computed = grad(q)
        dq_fd = approx_fprime(q, f, grad_precision)

        if verbose:
            print('dq_fd: ', np.round(dq_fd, decimals=4))
            print('dq_computed: ', np.round(dq_computed, decimals=4))

        npt.assert_allclose(dq_fd, dq_computed, rtol=rel_tol, atol=abs_tol)

    def test_dl_dq_eq_large(self, verbose=False):
        n, m = 100, 120

        prob = self.get_prob(n=n, m=m, P_scale=1.0, A_scale=1.0)
        P, q, A, l, u, true_x, true_y = prob

        l[20:40] = -osqp.constant('OSQP_INFTY', algebra='builtin')
        u[:20] = l[:20]

        def grad(q):
            _, dq, _, _, _ = self.get_grads(P, q, A, l, u, true_x)
            return dq

        def f(q):
            m = osqp.OSQP(algebra='builtin')
            m.setup(
                P,
                q,
                A,
                l,
                u,
                eps_abs=eps_abs,
                eps_rel=eps_rel,
                max_iter=max_iter,
                verbose=False,
            )
            res = m.solve()
            if res.info.status_val != osqp.SolverStatus.OSQP_SOLVED:
                raise ValueError('Problem not solved!')
            x_hat = res.x

            return 0.5 * np.sum(np.square(x_hat - true_x))

        dq_computed = grad(q)
        dq_fd = approx_fprime(q, f, grad_precision)

        if verbose:
            print('dq_fd: ', np.round(dq_fd, decimals=4))
            print('dq_computed: ', np.round(dq_computed, decimals=4))

        npt.assert_allclose(dq_fd, dq_computed, rtol=rel_tol_relaxed, atol=abs_tol_relaxed)

    def _test_dl_dq_nonzero_dy(self, verbose=False):
        n, m = 6, 3

        prob = self.get_prob(n=n, m=m, P_scale=1.0, A_scale=1.0)
        P, q, A, l, u, true_x, true_y = prob
        # u = l
        # l[20:40] = -osqp.constant('OSQP_INFTY', algebra='builtin')
        num_eq = 2
        u[:num_eq] = l[:num_eq]

        def grad(q):
            _, dq, _, _, _ = self.get_grads(P, q, A, l, u, true_x, true_y)
            return dq

        def f(q):
            m = osqp.OSQP(algebra='builtin')
            m.setup(
                P,
                q,
                A,
                l,
                u,
                eps_abs=eps_abs,
                eps_rel=eps_rel,
                max_iter=max_iter,
                verbose=False,
            )
            res = m.solve()
            if res.info.status_val != osqp.SolverStatus.OSQP_SOLVED:
                raise ValueError('Problem not solved!')
            x_hat = res.x
            y_hat = res.y
            yu_hat = np.maximum(y_hat, 0)
            yl_hat = -np.minimum(y_hat, 0)

            true_yu = np.maximum(true_y, 0)
            true_yl = -np.minimum(true_y, 0)
            # return 0.5 * np.sum(np.square(x_hat - true_x)) + np.sum(yl_hat) + np.sum(yu_hat)
            return 0.5 * (
                np.sum(np.square(x_hat - true_x))
                + np.sum(np.square(yl_hat - true_yl))
                + np.sum(np.square(yu_hat - true_yu))
            )

        dq_computed = grad(q)
        dq_fd = approx_fprime(q, f, grad_precision)

        if verbose:
            print('dq_fd: ', np.round(dq_fd, decimals=4))
            print('dq_computed: ', np.round(dq_computed, decimals=4))

        npt.assert_allclose(dq_fd, dq_computed, rtol=rel_tol, atol=abs_tol)
