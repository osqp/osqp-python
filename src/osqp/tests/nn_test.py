import numpy.random as npr
import numpy as np
import torch
import numpy.testing as npt
import scipy.sparse as spa
from scipy.optimize import approx_fprime
import pytest

import osqp
from osqp.nn.torch import OSQP

ATOL = 1e-2
RTOL = 1e-4
EPS = 1e-5

cuda = False
verbose = True


# Note (02/13/24)
# Some versions of Python/torch/numpy cannot coexist on certain platforms.
# This is a problem seen with numpy>=2.
# Support is gradually being added. Rather than keep track of which versions
# are supported and on what platforms (which is likely to change frequently),
# we do an early check that is seen to raise RuntimeError in these cases,
# and skip testing this module entirely.
try:
    torch.ones(1).cpu().numpy()
except RuntimeError:
    pytest.skip('torch/numpy mutual incompatibility', allow_module_level=True)


def get_grads(
    n_batch=1,
    n=10,
    m=3,
    P_scale=1.0,
    A_scale=1.0,
    u_scale=1.0,
    l_scale=1.0,
    algebra=None,
    solver_type=None,
):
    assert n_batch == 1
    npr.seed(1)
    L = np.random.randn(n, n)
    P = spa.csc_matrix(P_scale * L.dot(L.T))
    x_0 = npr.randn(n)
    s_0 = npr.rand(m)
    A = spa.csc_matrix(A_scale * npr.randn(m, n))
    u = A.dot(x_0) + A_scale * s_0
    l = -10 * A_scale * npr.rand(m)
    q = npr.randn(n)
    true_x = npr.randn(n)

    P, q, A, l, u, true_x = [x.astype(np.float64) for x in [P, q, A, l, u, true_x]]

    grads = get_grads_torch(P, q, A, l, u, true_x, algebra, solver_type)
    return [P, q, A, l, u, true_x], grads


def get_grads_torch(P, q, A, l, u, true_x, algebra, solver_type):
    P_idx = P.nonzero()
    P_shape = P.shape
    A_idx = A.nonzero()
    A_shape = A.shape

    P_torch, q_torch, A_torch, l_torch, u_torch, true_x_torch = [
        torch.DoubleTensor(x) if len(x) > 0 else torch.DoubleTensor() for x in [P.data, q, A.data, l, u, true_x]
    ]
    if cuda:
        P_torch, q_torch, A_torch, l_torch, u_torch, true_x_torch = [
            x.cuda() for x in [P.data, q, A.data, l, u, true_x]
        ]

    for x in [P_torch, q_torch, A_torch, l_torch, u_torch]:
        x.requires_grad = True

    x_hats = OSQP(
        P_idx,
        P_shape,
        A_idx,
        A_shape,
        algebra=algebra,
        solver_type=solver_type,
    )(P_torch, q_torch, A_torch, l_torch, u_torch)

    dl_dxhat = x_hats.data - true_x_torch
    x_hats.backward(dl_dxhat)

    grads = [x.grad.data.squeeze(0).cpu().numpy() for x in [P_torch, q_torch, A_torch, l_torch, u_torch]]
    return grads


def test_dl_dq(algebra, solver_type, atol, rtol, decimal_tol):
    n, m = 5, 5

    model = osqp.OSQP(algebra=algebra)
    if not model.has_capability('OSQP_CAPABILITY_DERIVATIVES'):
        pytest.skip('No derivatives capability')

    [P, q, A, l, u, true_x], [dP, dq, dA, dl, du] = get_grads(
        n=n,
        m=m,
        P_scale=100.0,
        A_scale=100.0,
        algebra=algebra,
        solver_type=solver_type,
    )

    def f(q):
        model.setup(P, q, A, l, u, solver_type=solver_type, verbose=False)
        res = model.solve()
        x_hat = res.x

        return 0.5 * np.sum(np.square(x_hat - true_x))

    dq_fd = approx_fprime(q, f, epsilon=EPS)
    if verbose:
        print('dq_fd: ', np.round(dq_fd, decimals=4))
        print('dq: ', np.round(dq, decimals=4))
    npt.assert_allclose(dq_fd, dq, rtol=RTOL, atol=ATOL)


def test_dl_dP(algebra, solver_type, atol, rtol, decimal_tol):
    n, m = 5, 5

    model = osqp.OSQP(algebra=algebra)
    if not model.has_capability('OSQP_CAPABILITY_DERIVATIVES'):
        pytest.skip('No derivatives capability')

    [P, q, A, l, u, true_x], [dP, dq, dA, dl, du] = get_grads(
        n=n,
        m=m,
        P_scale=100.0,
        A_scale=100.0,
        algebra=algebra,
        solver_type=solver_type,
    )

    def f(P):
        P = P.reshape(n, n)
        P = spa.csc_matrix(P)
        model.setup(P, q, A, l, u, solver_type=solver_type, verbose=False)
        res = model.solve()
        x_hat = res.x

        return 0.5 * np.sum(np.square(x_hat - true_x))

    dP_fd = approx_fprime(P.toarray().flatten(), f, epsilon=EPS)
    if verbose:
        print('dP_fd: ', np.round(dP_fd, decimals=4))
        print('dP: ', np.round(dP, decimals=4))
    npt.assert_allclose(dP_fd, dP, rtol=RTOL, atol=ATOL)


def test_dl_dA(algebra, solver_type, atol, rtol, decimal_tol):
    n, m = 5, 5

    model = osqp.OSQP(algebra=algebra)
    if not model.has_capability('OSQP_CAPABILITY_DERIVATIVES'):
        pytest.skip('No derivatives capability')

    [P, q, A, l, u, true_x], [dP, dq, dA, dl, du] = get_grads(
        n=n,
        m=m,
        P_scale=100.0,
        A_scale=100.0,
        algebra=algebra,
        solver_type=solver_type,
    )

    def f(A):
        A = A.reshape((m, n))
        A = spa.csc_matrix(A)
        model.setup(P, q, A, l, u, solver_type=solver_type, verbose=False)
        res = model.solve()
        x_hat = res.x

        return 0.5 * np.sum(np.square(x_hat - true_x))

    dA_fd = approx_fprime(A.toarray().flatten(), f, epsilon=EPS)
    if verbose:
        print('dA_fd: ', np.round(dA_fd, decimals=4))
        print('dA: ', np.round(dA, decimals=4))
    npt.assert_allclose(dA_fd, dA, rtol=RTOL, atol=ATOL)


def test_dl_dl(algebra, solver_type, atol, rtol, decimal_tol):
    n, m = 5, 5

    model = osqp.OSQP(algebra=algebra)
    if not model.has_capability('OSQP_CAPABILITY_DERIVATIVES'):
        pytest.skip('No derivatives capability')

    [P, q, A, l, u, true_x], [dP, dq, dA, dl, du] = get_grads(
        n=n,
        m=m,
        P_scale=100.0,
        A_scale=100.0,
        algebra=algebra,
        solver_type=solver_type,
    )

    def f(l):
        model.setup(P, q, A, l, u, solver_type=solver_type, verbose=False)
        res = model.solve()
        x_hat = res.x

        return 0.5 * np.sum(np.square(x_hat - true_x))

    dl_fd = approx_fprime(l, f, epsilon=EPS)
    if verbose:
        print('dl_fd: ', np.round(dl_fd, decimals=4))
        print('dl: ', np.round(dl, decimals=4))
    npt.assert_allclose(dl_fd, dl, rtol=RTOL, atol=ATOL)


def test_dl_du(algebra, solver_type, atol, rtol, decimal_tol):
    n, m = 5, 5

    model = osqp.OSQP(algebra=algebra)
    if not model.has_capability('OSQP_CAPABILITY_DERIVATIVES'):
        pytest.skip('No derivatives capability')

    [P, q, A, l, u, true_x], [dP, dq, dA, dl, du] = get_grads(
        n=n,
        m=m,
        P_scale=100.0,
        A_scale=100.0,
        algebra=algebra,
        solver_type=solver_type,
    )

    def f(u):
        model.setup(P, q, A, l, u, solver_type=solver_type, verbose=False)
        res = model.solve()
        x_hat = res.x

        return 0.5 * np.sum(np.square(x_hat - true_x))

    du_fd = approx_fprime(u, f, epsilon=EPS)
    if verbose:
        print('du_fd: ', np.round(du_fd, decimals=4))
        print('du: ', np.round(du, decimals=4))
    npt.assert_allclose(du_fd, du, rtol=RTOL, atol=ATOL)
