#  import cvxpy as cp
import numpy as np
import scipy.sparse as spa


def solve_high_accuracy(P, q, A, l, u):


    # Form QP in the form
    # minimize     0.5 x' * P * x + q' * x
    # subject to   Gx <= h
    #              Ax = b
    # Note P and q are the same

    # Divide in equalities and inequalities
    eq_idx = np.abs(u - l) <= 1e-7
    A_eq = A[eq_idx]
    b_eq = u[eq_idx]
    A_ineq = A[~eq_idx]
    u_ineq = u[~eq_idx]
    l_ineq = l[~eq_idx]

    # Construct QP
    P_qp = P
    q_qp = q
    A_qp = A_eq
    b_qp = b_eq
    G_qp = spa.vstack([A_ineq, -A_ineq], format='csc')
    h_qp = spa.hstack([u_ineq, -l_ineq])


    #



    x = cp.Variable(q.shape)
    if P.nnz == 0:
        quad_form = 0
    else:
        quad_form = .5 * cp.quad_form(x, P)

    if A.nnz == 0:
        constraints = []
    else:
        constraints = [A @ x <= u, l <= A @ x]

    problem = cp.Problem(cp.Minimize(quad_form + q @ x),
                         constraints)
    obj = problem.solve(solver=cp.ECOS,
                        abstol=1e-10, reltol=1e-10,
                        feastol=1e-10)
    x_val = x.value

    if A.nnz == 0:
        y_val = None
    else:
        y_val = problem.constraints[0].dual_value - \
            problem.constraints[1].dual_value

    return x_val, y_val, obj
