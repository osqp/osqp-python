import cvxpy as cp

rel_tol = 1e-03
abs_tol = 1e-04
decimal_tol = 5


def solve_high_accuracy(P, q, A, l, u):
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
