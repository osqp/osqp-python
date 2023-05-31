import numpy.random as npr
import numpy as np
import torch
import numdifftools as nd
import numpy.testing as npt
import scipy.sparse as spa
import pytest

import osqp
from osqp.nn.torch import OSQP

ATOL = 1e-2
RTOL = 1e-4

cuda = False
verbose = True


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


def test_dl_dp(algebra, solver_type, atol, rtol, decimal_tol):
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

    dq_fd = nd.Gradient(f)(q)
    if verbose:
        print('dq_fd: ', np.round(dq_fd, decimals=4))
        print('dq: ', np.round(dq, decimals=4))
    npt.assert_allclose(dq_fd, dq, rtol=RTOL, atol=ATOL)
