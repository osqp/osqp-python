import sys
import numpy as np
from scipy import sparse

from osqp.ext import OSQPSettings, CSC, OSQPSolver
from osqp.utils import prepare_data


def test_csc():
    settings = OSQPSettings()
    assert settings.rho == 0.1

    # P = sparse.diags([11., 0.], format='csc')
    # q = np.array([3, 4])
    # A = sparse.csc_matrix(
    #     [[-1, 0], [0, -1], [-1, -3], [2, 5], [3, 4]])
    # u = np.array([0., 0., -15, 100, 80])
    # l = -1e06 * np.ones(len(u))
    #
    # (n, m), P_x, P_i, P_p, q, A_x, A_i, A_p, l, u = prepare_data(P, q, A, l, u)[0]
    #
    # A = CSC(m, n, len(A_i), A_x, A_i, A_p)
    # P = CSC(n, n, len(P_i), P_x, P_i, P_p)
    #
    # solver = OSQPSolver(P, q, A, l, u, m, n, settings)
    # info = solver.solve()
    # print(info)

