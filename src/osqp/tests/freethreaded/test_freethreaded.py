"""
Tests that expose thread-safety issues in the OSQP extension module when running
under free-threaded Python 3.13t with the GIL disabled.

Run with:
    /path/to/python3.13t -Xgil=0 -m pytest src/osqp/tests/test_freethreaded.py -v

Under regular Python (GIL enabled), most tests pass because the GIL serializes
C extension calls.  Under free-threaded Python (-Xgil=0), the races manifest as
either crashes or result-consistency failures.

Issues being tested (see src/bindings.cpp.in):
  1. get_solution(): unsynchronized check-then-act on _solution_cache (lines 168-171)
  2. PyOSQPSolution holds OSQPSolution& — a raw reference into solver-owned memory (line 74)
  3. solve(), update_data_vec(), etc. access _solver with no mutex (lines 196-274)
  4. CSC stores raw pointers into numpy array buffers (lines 44-46)
"""
import gc
import importlib
import sys
import threading

import numpy as np
import numpy.testing as npt
import scipy.sparse as spa
import pytest

import osqp

GIL_DISABLED = not getattr(sys, '_is_gil_enabled', lambda: True)()

pytestmark = pytest.mark.skipif(
    not osqp.algebra_available('builtin'),
    reason='Builtin algebra not available',
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_problem():
    P = spa.diags([11.0, 0.0], format='csc', dtype=np.float64)
    q = np.array([3.0, 4.0])
    A = spa.csc_matrix(
        np.array([[-1.0, 0.0], [0.0, -1.0], [-1.0, -3.0], [2.0, 5.0], [3.0, 4.0]])
    )
    u = np.array([0.0, 0.0, -15.0, 100.0, 80.0])
    l = -1e6 * np.ones(5)
    return P, q, A, l, u


def _make_osqp_solver():
    m = osqp.OSQP(algebra='builtin')
    P, q, A, l, u = _make_problem()
    m.setup(P, q, A, l, u, verbose=False, eps_abs=1e-5, eps_rel=1e-5)
    return m


def _reference_x():
    s = _make_osqp_solver()
    s._solver.solve()
    return s._solver.solution.x.copy()


# ---------------------------------------------------------------------------
# Issue 1: race on _solution_cache in get_solution()
#
#   if (!_solution_cache) {                                    // check
#       _solution_cache = make_unique<PyOSQPSolution>(...);   // act
#   }
#   return *_solution_cache;
#
# N threads can all pass the null-check before any of them completes the
# assignment, each creating a separate PyOSQPSolution.  The last assignment
# wins and destroys the previous objects — threads that already captured a
# pointer to a destroyed object dereference freed memory.
# ---------------------------------------------------------------------------

class TestGetSolutionRace:

    def test_concurrent_first_access_is_consistent(self):
        """
        All N threads race to be the first caller of .solution on a freshly
        solved solver whose _solution_cache is still null.  Under a data race
        some threads get an x vector from a destroyed PyOSQPSolution; the
        resulting values will differ from the reference solution.
        """
        solver = _make_osqp_solver()
        solver._solver.solve()
        # _solution_cache is STILL null here — .solution has not been called yet.

        N = 40
        barrier = threading.Barrier(N)
        xs = []
        errors = []
        lock = threading.Lock()

        def worker():
            try:
                barrier.wait()           # start all threads simultaneously
                x = solver._solver.solution.x.copy()
                with lock:
                    xs.append(x)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f'{len(errors)} threads raised exceptions: {errors[0]}'
        assert len(xs) == N

        ref = xs[0]
        for i, x in enumerate(xs[1:], 1):
            npt.assert_allclose(
                x, ref, rtol=1e-4, atol=1e-6,
                err_msg=(
                    f'Thread {i} got a different x than thread 0 — '
                    'concurrent get_solution() data race detected'
                ),
            )


# ---------------------------------------------------------------------------
# Issue 2: dangling reference — solution outlives solver
#
#   .def_property_readonly("solution", &PyOSQPSolver::get_solution,
#                          py::return_value_policy::reference)
#
# py::return_value_policy::reference does NOT keep the parent (PyOSQPSolver)
# alive when a child (PyOSQPSolution) Python object is held.  When the solver
# is garbage-collected, _solution_cache is destroyed and the C++ reference
# inside PyOSQPSolution becomes dangling.
# ---------------------------------------------------------------------------

class TestSolutionDanglingReference:

    def test_solution_access_after_solver_gc(self):
        """
        Obtain a direct reference to the low-level PyOSQPSolution object, then
        destroy the solver.  Accessing .x afterwards reads through a dangling
        C++ reference (OSQPSolution& _solution) — undefined behaviour.

        Expected outcome: either a segfault / SystemError (UB caught by the OS
        or Python runtime) or silently wrong values.  The test marks this path
        as a known bug by calling pytest.fail() if we reach the .x access
        without an exception, because that means we're reading freed memory.
        """
        solver = _make_osqp_solver()
        solver._solver.solve()

        # py::return_value_policy::reference — no keep-alive on the parent
        low_level_solution = solver._solver.solution

        # Release all references to the solver so its C++ destructor runs
        del solver
        gc.collect()

        # Any access here is a use-after-free
        try:
            x = low_level_solution.x
        except Exception:
            # Getting an exception means Python/C++ caught the problem
            return

        pytest.fail(
            f'Read solution.x={x} after solver destruction — '
            'this is a use-after-free / dangling C++ reference (OSQPSolution& _solution)'
        )


# ---------------------------------------------------------------------------
# Issue 3a: concurrent solve() calls on the same solver
#
# All PyOSQPSolver methods access _solver (raw OSQPSolver*) with no mutex.
# The py::gil_scoped_release inside solve() used to prevent other Python
# threads from calling back into the solver during a solve.  In free-threaded
# Python that protection is gone.
# ---------------------------------------------------------------------------

class TestConcurrentSolve:

    def test_results_match_reference(self):
        """
        Serial solves of the same problem always give the same x.  Concurrent
        solves on the same solver object should also be consistent — but the
        lack of synchronization means OSQP's internal work arrays get
        corrupted, producing wrong results.
        """
        solver = _make_osqp_solver()
        ref = _reference_x()

        N_THREADS = 8
        N_ITER = 30
        barrier = threading.Barrier(N_THREADS)
        results = []
        errors = []
        lock = threading.Lock()

        def worker():
            try:
                barrier.wait()
                for _ in range(N_ITER):
                    solver._solver.solve()
                    x = solver._solver.solution.x.copy()
                    with lock:
                        results.append(x)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            pytest.fail(f'Concurrent solve() raised {len(errors)} errors: {errors[0]}')

        for i, x in enumerate(results):
            npt.assert_allclose(
                x, ref, rtol=1e-3, atol=1e-3,
                err_msg=(
                    f'Result #{i} does not match reference — '
                    'concurrent solve() data race corrupted internal OSQP state'
                ),
            )


# ---------------------------------------------------------------------------
# Issue 3b: concurrent update_data_vec + solve
# ---------------------------------------------------------------------------

class TestConcurrentUpdateAndSolve:

    def test_no_crash_or_exception(self):
        """
        update_data_vec() writes q/l/u into _solver while solve() is reading
        them.  Interleaving from separate threads with no synchronization means
        solve() can read a half-updated cost vector, corrupting the iterate.
        Even if no exception is raised, silent data corruption is occurring.
        """
        solver = _make_osqp_solver()
        P, q, A, l, u = _make_problem()

        errors = []
        stop = threading.Event()
        N = 150

        def solve_loop():
            for _ in range(N):
                if stop.is_set():
                    break
                try:
                    solver._solver.solve()
                except Exception as exc:
                    errors.append(('solve', exc))
                    stop.set()
                    break

        def update_loop():
            for i in range(N):
                if stop.is_set():
                    break
                try:
                    q_new = (q + i * 0.01).astype(np.float64)
                    solver._solver.update_data_vec(q_new, None, None)
                except Exception as exc:
                    errors.append(('update', exc))
                    stop.set()
                    break

        t1 = threading.Thread(target=solve_loop)
        t2 = threading.Thread(target=update_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        if errors:
            pytest.fail(
                f'Concurrent update+solve produced {len(errors)} errors: {errors[0]}'
            )


# ---------------------------------------------------------------------------
# Issue 4: CSC raw pointer aliasing
#
#   this->_csc->p = (OSQPInt *)this->_p.data();   // raw pointer into numpy
#   this->_csc->i = (OSQPInt *)this->_i.data();
#   this->_csc->x = (OSQPFloat *)this->_x.data();
#
# The OSQPCscMatrix struct inside CSC holds pointers directly into the numpy
# array buffers.  If a second thread mutates those arrays while solve() is
# reading through _csc->x, the solver reads a torn/mixed data set.
# ---------------------------------------------------------------------------

class TestCSCNumpyAliasing:

    def test_concurrent_mutation_and_solve(self):
        """
        One thread mutates the numpy array that backs P_csc.x while another
        thread calls solve().  Because _csc->x is a raw alias into that same
        numpy buffer (verified: P_sp.data.ctypes.data == P_csc.x.ctypes.data),
        the solver reads whatever value the mutator last wrote — a data race
        that TSAN would flag on the underlying memory.
        """
        ext = importlib.import_module('osqp.ext_builtin')
        P, q, A, l, u = _make_problem()

        P_sp = P.astype(np.float64)   # keep the scipy matrix alive so .data stays valid
        P_csc = ext.CSC(P_sp)         # _csc->x aliases P_sp.data's buffer
        A_csc = ext.CSC(A.astype(np.float64))

        settings = ext.OSQPSettings()
        ext.osqp_set_default_settings(settings)
        settings.verbose = False

        raw_solver = ext.OSQPSolver(
            P_csc,
            q.astype(np.float64),
            A_csc,
            l.astype(np.float64),
            u.astype(np.float64),
            5,
            2,
            settings,
        )

        errors = []
        stop = threading.Event()

        def mutate_thread():
            try:
                for _ in range(500):
                    if stop.is_set():
                        break
                    # P_sp.data and P_csc.x share the same buffer;
                    # writing here races with solve()'s read via _csc->x.
                    P_sp.data[0] = np.random.rand() * 22.0
            except Exception as exc:
                errors.append(('mutate', exc))

        def solve_thread():
            try:
                for _ in range(100):
                    if stop.is_set():
                        break
                    raw_solver.solve()
            except Exception as exc:
                errors.append(('solve', exc))
                stop.set()

        t_solve = threading.Thread(target=solve_thread)
        t_mutate = threading.Thread(target=mutate_thread)
        t_solve.start()
        t_mutate.start()
        t_solve.join()
        t_mutate.join()

        if errors:
            pytest.fail(
                f'CSC aliasing race produced {len(errors)} errors: {errors[0]}'
            )
