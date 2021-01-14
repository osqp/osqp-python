#  import cvxpy as cp
import numpy as np
import scipy.sparse as spa
from cvxopt import spmatrix, matrix, solvers

rel_tol = 1e-03
abs_tol = 1e-04
decimal_tol = 4


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
    P_qp = P.tocoo().astype(np.double)
    q_qp = q.astype(np.double)
    A_qp = A_eq.tocoo().astype(np.double)
    b_qp = b_eq.astype(np.double)
    G_qp = spa.vstack([A_ineq, -A_ineq], format='coo').astype(np.double)
    h_qp = np.hstack([u_ineq, -l_ineq]).astype(np.double)

    # Construct and solve CVXOPT problem
    P_cvxopt = spmatrix(P_qp.data, P_qp.row, P_qp.col, P_qp.shape, tc='d')
    q_cvxopt = matrix(q_qp, tc='d')
    A_cvxopt = spmatrix(A_qp.data, A_qp.row, A_qp.col, A_qp.shape, tc='d')
    b_cvxopt = matrix(b_qp, tc='d')
    G_cvxopt = spmatrix(G_qp.data, G_qp.row, G_qp.col, G_qp.shape, tc='d')
    h_cvxopt = matrix(h_qp, tc='d')

    solvers.options['show_progress'] = False
    solution = solvers.qp(P_cvxopt, q_cvxopt,
                          G_cvxopt, h_cvxopt, A_cvxopt, b_cvxopt)

    # Recover primal variable and solution
    x_val = np.array(solution['x']).flatten()
    obj = solution['primal objective']

    # Recover dual variables
    dual_eq = np.array(solution['y']).flatten()
    dual_ineq_up = np.array(solution['z']).flatten()[:int(len(h_qp)/2)]
    dual_ineq_low = np.array(solution['z']).flatten()[int(len(h_qp)/2):]
    y_val = np.zeros(len(u))
    y_val[eq_idx] = dual_eq
    y_val[~eq_idx] = dual_ineq_up - dual_ineq_low

    return x_val, y_val, obj
