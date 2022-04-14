# Test osqp python module
import osqp
# import osqppurepy as osqp
import numpy as np
from scipy import sparse

# Unit Test
import unittest
import pytest
import numpy.testing as nptest
import shutil as sh
import sys


class codegen_matrices_tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        P = sparse.diags([11., 0.1], format='csc')
        P_new = sparse.eye(2, format='csc')
        q = np.array([3, 4])
        A = sparse.csc_matrix([[-1, 0], [0, -1], [-1, -3],
                                    [2, 5], [3, 4]])
        A_new = sparse.csc_matrix([[-1, 0], [0, -1], [-2, -2],
                                        [2, 5], [3, 4]])
        u = np.array([0, 0, -15, 100, 80])
        l = -np.inf * np.ones(len(u))
        n = P.shape[0]
        m = A.shape[0]
        opts = {'verbose': False,
                     'eps_abs': 1e-08,
                     'eps_rel': 1e-08,
                     'alpha': 1.6,
                     'max_iter': 3000,
                     'warm_start': True}

        model = osqp.OSQP()
        model.setup(P=P, q=q, A=A, l=l, u=u, **opts)

        model_dir = model.codegen('code2', python_ext_name='mat_emosqp', force_rewrite=True, parameters='matrices')
        sh.rmtree('code2')
        sys.path.append(model_dir)

        cls.m = m
        cls.n = n
        cls.P = P
        cls.P_new = P_new
        cls.q = q
        cls.A = A
        cls.A_new = A_new
        cls.l = l
        cls.u = u
        cls.opts = opts

    def setUp(self):

        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
                         **self.opts)

    def test_solve(self):
        import mat_emosqp

        # Solve problem
        x, y, _, _, _ = mat_emosqp.solve()

        # Assert close
        nptest.assert_array_almost_equal(x, np.array([0., 5.]), decimal=5)
        nptest.assert_array_almost_equal(
            y, np.array([1.5, 0., 1.5, 0., 0.]), decimal=5)

    def test_update_P(self):
        import mat_emosqp

        # Update matrix P
        Px = self.P_new.data
        Px_idx = np.arange(self.P_new.nnz)
        mat_emosqp.update_P(Px, Px_idx, len(Px))

        # Solve problem
        x, y, _, _, _ = mat_emosqp.solve()

        # Assert close
        nptest.assert_array_almost_equal(x, np.array([0., 5.]), decimal=5)
        nptest.assert_array_almost_equal(
            y, np.array([0., 0., 3., 0., 0.]), decimal=5)

        # Update matrix P to the original value
        Px = self.P.data
        Px_idx = np.arange(self.P.nnz)
        mat_emosqp.update_P(Px, Px_idx, len(Px))

    def test_update_P_allind(self):
        import mat_emosqp

        # Update matrix P
        Px = self.P_new.data
        mat_emosqp.update_P(Px, None, 0)
        x, y, _, _, _ = mat_emosqp.solve()

        # Assert close
        nptest.assert_array_almost_equal(x, np.array([0., 5.]), decimal=5)
        nptest.assert_array_almost_equal(
            y, np.array([0., 0., 3., 0., 0.]), decimal=5)

        # Update matrix P to the original value
        Px_idx = np.arange(self.P.nnz)
        mat_emosqp.update_P(Px, Px_idx, len(Px))

    def test_update_A(self):
        import mat_emosqp

        # Update matrix A
        Ax = self.A_new.data
        Ax_idx = np.arange(self.A_new.nnz)
        mat_emosqp.update_A(Ax, Ax_idx, len(Ax))

        # Solve problem
        x, y, _, _, _ = mat_emosqp.solve()

        # Assert close
        nptest.assert_array_almost_equal(x,
                                         np.array([0.15765766, 7.34234234]), decimal=5)
        nptest.assert_array_almost_equal(
            y, np.array([0., 0., 2.36711712, 0., 0.]), decimal=5)

        # Update matrix A to the original value
        Ax = self.A.data
        Ax_idx = np.arange(self.A.nnz)
        mat_emosqp.update_A(Ax, Ax_idx, len(Ax))

    def test_update_A_allind(self):
        import mat_emosqp

        # Update matrix A
        Ax = self.A_new.data
        mat_emosqp.update_A(Ax, None, 0)
        x, y, _, _, _ = mat_emosqp.solve()

        # Assert close
        nptest.assert_array_almost_equal(x,
                                         np.array([0.15765766, 7.34234234]), decimal=5)
        nptest.assert_array_almost_equal(
            y, np.array([0., 0., 2.36711712, 0., 0.]), decimal=5)

        # Update matrix A to the original value
        Ax = self.A.data
        Ax_idx = np.arange(self.A.nnz)
        mat_emosqp.update_A(Ax, Ax_idx, len(Ax))

    def test_update_P_A_indP_indA(self):
        import mat_emosqp

        # Update matrices P and A
        Px = self.P_new.data
        Px_idx = np.arange(self.P_new.nnz)
        Ax = self.A_new.data
        Ax_idx = np.arange(self.A_new.nnz)
        mat_emosqp.update_P_A(Px, Px_idx, len(Px), Ax, Ax_idx, len(Ax))

        # Solve problem
        x, y, _, _, _ = mat_emosqp.solve()

        # Assert close
        nptest.assert_array_almost_equal(x, np.array([4.25, 3.25]), decimal=5)
        nptest.assert_array_almost_equal(
            y, np.array([0., 0., 3.625, 0., 0.]), decimal=5)

        # Update matrices P and A to the original values
        Px = self.P.data
        Ax = self.A.data
        mat_emosqp.update_P_A(Px, None, 0, Ax, None, 0)

    def test_update_P_A_indP(self):
        import mat_emosqp

        # Update matrices P and A
        Px = self.P_new.data
        Px_idx = np.arange(self.P_new.nnz)
        Ax = self.A_new.data
        mat_emosqp.update_P_A(Px, Px_idx, len(Px), Ax, None, 0)
        x, y, _, _, _ = mat_emosqp.solve()

        # Assert close
        nptest.assert_array_almost_equal(x, np.array([4.25, 3.25]), decimal=5)
        nptest.assert_array_almost_equal(
            y, np.array([0., 0., 3.625, 0., 0.]), decimal=5)

        # Update matrices P and A to the original values
        Px = self.P.data
        Ax = self.A.data
        mat_emosqp.update_P_A(Px, None, 0, Ax, None, 0)

    def test_update_P_A_indA(self):
        import mat_emosqp

        # Update matrices P and A
        Px = self.P_new.data
        Ax = self.A_new.data
        Ax_idx = np.arange(self.A_new.nnz)
        mat_emosqp.update_P_A(Px, None, 0, Ax, Ax_idx, len(Ax))
        x, y, _, _, _ = mat_emosqp.solve()

        # Assert close
        nptest.assert_array_almost_equal(x, np.array([4.25, 3.25]), decimal=5)
        nptest.assert_array_almost_equal(
            y, np.array([0., 0., 3.625, 0., 0.]), decimal=5)

        # Update matrix P to the original value
        Px = self.P.data
        Px_idx = np.arange(self.P.nnz)
        Ax = self.A.data
        Ax_idx = np.arange(self.A.nnz)
        mat_emosqp.update_P_A(Px, Px_idx, len(Px), Ax, Ax_idx, len(Ax))

    def test_update_P_A_allind(self):
        import mat_emosqp

        # Update matrices P and A
        Px = self.P_new.data
        Ax = self.A_new.data
        mat_emosqp.update_P_A(Px, None, 0, Ax, None, 0)
        x, y, _, _, _ = mat_emosqp.solve()

        # Assert close
        nptest.assert_array_almost_equal(x, np.array([4.25, 3.25]), decimal=5)
        nptest.assert_array_almost_equal(
            y, np.array([0., 0., 3.625, 0., 0.]), decimal=5)

        # Update matrices P and A to the original values
        Px = self.P.data
        Ax = self.A.data
        mat_emosqp.update_P_A(Px, None, 0, Ax, None, 0)
