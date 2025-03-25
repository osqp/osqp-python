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

    # Setup workspace
    prob.setup(P, q, A, l, u)

    # Solve problem
    res = prob.solve()

    # Update problem
    q_new = np.array([2, 3])
    l_new = np.array([2, -1, -1])
    u_new = np.array([2, 2.5, 2.5])
    prob.update(q=q_new, l=l_new, u=u_new)

    # Solve updated problem
    res = prob.solve()

    print('Status:', res.info.status)
    print('Objective value:', res.info.obj_val)
    print('Optimal solution x:', res.x)
