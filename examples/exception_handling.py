import osqp
import numpy as np
from scipy import sparse


"""

`osqp.OSQPException`s might be raised during `.setup()`, `.update_settings()`,
or `.solve()`. This example demonstrates how to catch an `osqp.OSQPException`
raised during `.setup()`, and how to compare it to a specific `osqp.SolverError`.

Exceptions other than `osqp.OSQPException` might also be raised, but these
are typically errors in using the wrapper, and are not raised by the underlying
`osqp` library itself.

"""

if __name__ == '__main__':

    P = sparse.triu([[2.0, 5.0], [5.0, 1.0]], format='csc')
    q = np.array([3.0, 4.0])
    A = sparse.csc_matrix([[-1.0, 0.0], [0.0, -1.0], [-1.0, 3.0], [2.0, 5.0], [3.0, 4]])
    l = -np.inf * np.ones(A.shape[0])
    u = np.array([0.0, 0.0, -15.0, 100.0, 80.0])

    prob = osqp.OSQP()

    try:
        prob.setup(P, q, A, l, u)
    except osqp.OSQPException as e:
        # Our problem is non-convex, so we get a osqp.OSQPException
        # during .setup()
        assert e == osqp.SolverError.OSQP_NONCVX_ERROR
