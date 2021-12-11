import numpy as np
from scipy import sparse

from osqp.ext import CSC, OSQPSettings, OSQPSolver


def test_solver():
    settings = OSQPSettings()
    P = CSC(sparse.diags([11., 0.], format='csc'))
    q = np.array([3, 4])
    A = CSC(sparse.csc_matrix(
        [[-1, 0], [0, -1], [-1, -3], [2, 5], [3, 4]]))
    u = np.array([0., 0., -15, 100, 80])
    l = -1e06 * np.ones(len(u))

    n = 2
    m = 5

    solver = OSQPSolver(P, q.astype(np.float64), A, l.astype(np.float64), u.astype(np.float64), m, n, settings)
    solver.solve()
