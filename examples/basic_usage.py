import osqp
import numpy as np
from scipy import sparse


if __name__ == '__main__':
    # Define problem data
    P = sparse.csc_matrix([[4, 1], [1, 2]])
    q = np.array([1, 1])
    A = sparse.csc_matrix([[1, 1], [1, 0], [0, 1]])
    l = np.array([1, 0, 0])
    u = np.array([1, 0.7, 0.7])

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace and change alpha parameter
    prob.setup(P, q, A, l, u, alpha=1.0)

    # Settings can be changed using .update_settings()
    prob.update_settings(polishing=1)

    # Solve problem
    res = prob.solve(raise_error=True)

    # Check solver status
    # For all values, see https://osqp.org/docs/interfaces/status_values.html
    assert res.info.status_val == osqp.SolverStatus.OSQP_SOLVED

    print('Status:', res.info.status)
    print('Objective value:', res.info.obj_val)
    print('Optimal solution x:', res.x)
