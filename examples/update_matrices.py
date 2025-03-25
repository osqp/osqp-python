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
    # IMPORTANT: The sparsity structure of P/A should remain the same,
    # so we only update Px and Ax
    # (i.e. the actual data values at indices with nonzero values)
    # NB: Update only upper triangular part of P
    P_new = sparse.csc_matrix([[5, 1.5], [1.5, 1]])
    A_new = sparse.csc_matrix([[1.2, 1.1], [1.5, 0], [0, 0.8]])
    prob.update(Px=sparse.triu(P_new).data, Ax=A_new.data)

    # Solve updated problem
    res = prob.solve()

    print('Status:', res.info.status)
    print('Objective value:', res.info.obj_val)
    print('Optimal solution x:', res.x)
