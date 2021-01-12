#  import cvxpy as cp
import numpy as np
import scipy.sparse as spa
from cvxopt import spmatrix, matrix, solvers

rel_tol = 1e-03
abs_tol = 1e-04
decimal_tol = 5


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
    P_qp = P.tocoo()
    q_qp = q
    A_qp = A_eq.tocoo()
    b_qp = b_eq
    G_qp = spa.vstack([A_ineq, -A_ineq], format='coo')
    h_qp = np.hstack([u_ineq, -l_ineq])

    # Construct CVXOPT problem
    P_cvxopt = spmatrix(P_qp.data.astype(float), P_qp.row, P_qp.col, P_qp.shape)
    q_cvxopt = matrix(np.reshape(q_qp.astype(float), (-1, 1)))
    if A_qp.nnz == 0:
        A_cvxopt = None
        b_cvxopt = None
    else:
        A_cvxopt = spmatrix(A_qp.data.astype(float), A_qp.row, A_qp.col, A_qp.shape)
        b_cvxopt = matrix(np.reshape(b_qp, (-1, 1)))
    if G_qp.nnz == 0:
        G_cvxopt = None
        h_cvxopt = None
    else:
        G_cvxopt = spmatrix(G_qp.data.astype(float), G_qp.row, G_qp.col, G_qp.shape)
        h_cvxopt = matrix(np.reshape(h_qp.astype(float), (-1, 1)))

    try:
        solution = solvers.qp(P_cvxopt, q_cvxopt, G_cvxopt, h_cvxopt, A_cvxopt, b_cvxopt)
    except:
        import ipdb; ipdb.set_trace()
    x_val = solution['x']

    #
    #
    #
    #
    #  x = cp.Variable(q.shape)
    #  if P.nnz == 0:
    #      quad_form = 0
    #  else:
    #      quad_form = .5 * cp.quad_form(x, P)
    #
    #  if A.nnz == 0:
    #      constraints = []
    #  else:
    #      constraints = [A @ x <= u, l <= A @ x]
    #
    #  problem = cp.Problem(cp.Minimize(quad_form + q @ x),
    #                       constraints)
    #  obj = problem.solve(solver=cp.ECOS,
    #                      abstol=1e-10, reltol=1e-10,
    #                      feastol=1e-10)
    #  x_val = x.value
    #
    #  if A.nnz == 0:
    #      y_val = None
    #  else:
    #      y_val = problem.constraints[0].dual_value - \
    #          problem.constraints[1].dual_value

    return x_val, y_val, obj
