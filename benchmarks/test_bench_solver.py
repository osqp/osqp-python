"""Benchmarks for the OSQP solver."""

import numpy as np
import pytest
from scipy import sparse

import osqp


def _make_small_qp():
    """Create a small 2-variable QP problem."""
    P = sparse.csc_matrix([[4.0, 1.0], [1.0, 2.0]])
    q = np.array([1.0, 1.0])
    A = sparse.csc_matrix([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    l = np.array([1.0, 0.0, 0.0])
    u = np.array([1.0, 0.7, 0.7])
    return P, q, A, l, u


def _make_medium_qp(n=50, seed=0):
    """Create a medium-sized random QP problem."""
    np.random.seed(seed)
    M = sparse.random(n, n, density=0.3, format='csc')
    P = (M.T @ M + 0.1 * sparse.eye(n)).tocsc()
    q = np.random.randn(n)
    m = 2 * n
    A = sparse.random(m, n, density=0.3, format='csc')
    l = -2.0 + np.random.randn(m)
    u = 2.0 + np.random.randn(m)
    return P, q, A, l, u


def _make_large_qp(n=200, seed=1):
    """Create a larger random QP problem."""
    np.random.seed(seed)
    M = sparse.random(n, n, density=0.1, format='csc')
    P = (M.T @ M + 0.5 * sparse.eye(n)).tocsc()
    q = np.random.randn(n)
    m = int(1.5 * n)
    A = sparse.random(m, n, density=0.15, format='csc')
    l = -3.0 + np.random.randn(m)
    u = 3.0 + np.random.randn(m)
    return P, q, A, l, u


# ---------------------------------------------------------------------------
# Setup benchmarks
# ---------------------------------------------------------------------------


def test_bench_setup_small(benchmark):
    """Benchmark solver setup for a small QP problem."""
    P, q, A, l, u = _make_small_qp()

    def _setup():
        solver = osqp.OSQP()
        solver.setup(P, q, A, l, u, verbose=False)

    benchmark(_setup)


def test_bench_setup_medium(benchmark):
    """Benchmark solver setup for a medium QP problem (n=50)."""
    P, q, A, l, u = _make_medium_qp()

    def _setup():
        solver = osqp.OSQP()
        solver.setup(P, q, A, l, u, verbose=False)

    benchmark(_setup)


def test_bench_setup_large(benchmark):
    """Benchmark solver setup for a large QP problem (n=200)."""
    P, q, A, l, u = _make_large_qp()

    def _setup():
        solver = osqp.OSQP()
        solver.setup(P, q, A, l, u, verbose=False)

    benchmark(_setup)


# ---------------------------------------------------------------------------
# Solve benchmarks
# ---------------------------------------------------------------------------


def test_bench_solve_small(benchmark):
    """Benchmark solve for a small QP problem."""
    P, q, A, l, u = _make_small_qp()
    solver = osqp.OSQP()
    solver.setup(P, q, A, l, u, verbose=False)

    benchmark(solver.solve)


def test_bench_solve_medium(benchmark):
    """Benchmark solve for a medium QP problem (n=50)."""
    P, q, A, l, u = _make_medium_qp()
    solver = osqp.OSQP()
    solver.setup(P, q, A, l, u, verbose=False, max_iter=4000)

    benchmark(solver.solve)


def test_bench_solve_large(benchmark):
    """Benchmark solve for a large QP problem (n=200)."""
    P, q, A, l, u = _make_large_qp()
    solver = osqp.OSQP()
    solver.setup(P, q, A, l, u, verbose=False, max_iter=4000)

    benchmark(solver.solve)


# ---------------------------------------------------------------------------
# Update benchmarks
# ---------------------------------------------------------------------------


def test_bench_update_vectors(benchmark):
    """Benchmark updating linear cost and bounds."""
    P, q, A, l, u = _make_medium_qp()
    solver = osqp.OSQP()
    solver.setup(P, q, A, l, u, verbose=False)

    np.random.seed(42)
    q_new = np.random.randn(len(q))
    l_new = -2.0 + np.random.randn(len(l))
    u_new = 2.0 + np.random.randn(len(u))

    def _update_and_solve():
        solver.update(q=q_new, l=l_new, u=u_new)
        solver.solve()

    benchmark(_update_and_solve)


# ---------------------------------------------------------------------------
# Setup + solve (end-to-end)
# ---------------------------------------------------------------------------


def test_bench_end_to_end_small(benchmark):
    """Benchmark full setup and solve cycle for a small QP problem."""
    P, q, A, l, u = _make_small_qp()

    def _setup_and_solve():
        solver = osqp.OSQP()
        solver.setup(P, q, A, l, u, verbose=False)
        solver.solve()

    benchmark(_setup_and_solve)


def test_bench_end_to_end_medium(benchmark):
    """Benchmark full setup and solve cycle for a medium QP problem (n=50)."""
    P, q, A, l, u = _make_medium_qp()

    def _setup_and_solve():
        solver = osqp.OSQP()
        solver.setup(P, q, A, l, u, verbose=False, max_iter=4000)
        solver.solve()

    benchmark(_setup_and_solve)


# ---------------------------------------------------------------------------
# Polishing benchmark
# ---------------------------------------------------------------------------


def test_bench_solve_with_polishing(benchmark):
    """Benchmark solve with polishing enabled."""
    P, q, A, l, u = _make_medium_qp()
    solver = osqp.OSQP()
    solver.setup(P, q, A, l, u, verbose=False, polishing=True, max_iter=4000)

    benchmark(solver.solve)


# ---------------------------------------------------------------------------
# Warm start benchmark
# ---------------------------------------------------------------------------


def test_bench_warm_start_solve(benchmark):
    """Benchmark solve with warm starting from a previous solution."""
    P, q, A, l, u = _make_medium_qp()
    solver = osqp.OSQP()
    solver.setup(P, q, A, l, u, verbose=False, warm_starting=True)

    # Solve once to get initial solution
    res = solver.solve()
    x0 = res.x
    y0 = res.y

    def _warm_start_and_solve():
        solver.warm_start(x=x0, y=y0)
        solver.solve()

    benchmark(_warm_start_and_solve)
