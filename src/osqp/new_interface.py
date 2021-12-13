import numpy as np
from osqp.ext import CSC, OSQPInfo, OSQPSolver, OSQPSettings, OSQPSolution
import osqp.utils as utils


class OSQP:
    def __init__(self, *args, **kwargs):
        self.m = None
        self.n = None
        self.P = None
        self.q = None
        self.A = None
        self.l = None
        self.u = None
        self.settings = OSQPSettings()

        self._solver = None

    def update_settings(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self.settings, k, v)

        if self._solver is not None:
            self._solver.update_settings(self.settings)

    def update(self, **kwargs):
        return self._solver.update_data_vec(
            q=kwargs.get('q'),
            l=kwargs.get('l'),
            u=kwargs.get('u')
        )

    def setup(self, P, q, A, l, u, **settings):
        self.m = l.shape[0]
        self.n = q.shape[0]
        self.P = CSC(P)
        self.q = q.astype(np.float64)
        self.A = CSC(A)
        self.l = l.astype(np.float64)
        self.u = u.astype(np.float64)

        self.update_settings(**settings)

        self._solver = OSQPSolver(self.P, self.q, self.A, self.l, self.u, self.m, self.n, self.settings)

    def solve(self):
        return self._solver.solve()  # (solution, info) 2-tuple
