from types import SimpleNamespace
import warnings
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

        # Some setting names have changed. Support the old names for now, but warn the caller.
        renamed_settings = {'polish': 'polishing', 'warm_start': 'warm_starting'}
        for k, v in renamed_settings.items():
            if k in kwargs:
                warnings.warn(f'"{k}" is deprecated. Please use "{v}" instead.', DeprecationWarning)
                kwargs[v] = kwargs[k]
                del kwargs[k]

        new_settings = OSQPSettings()
        for k in OSQPSettings.__dict__:
            if not k.startswith('__'):
                if k in kwargs:
                    setattr(new_settings, k, kwargs[k])
                else:
                    setattr(new_settings, k, getattr(self.settings, k))

        if self._solver is not None:
            if 'rho' in kwargs:
                self._solver.update_rho(kwargs.pop('rho'))
            if kwargs:
                self._solver.update_settings(new_settings)
            self.settings = self._solver.get_settings()  # TODO: Why isn't this just an attribute?
        else:
            self.settings = new_settings

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
        self._solver.solve()
        # TODO: The following structure is only to maintain backward compatibility, where x/y are attributes
        # directly inside the returned object on solve(). This should be simplified!
        return SimpleNamespace(
            x=self._solver.solution.x,
            y=self._solver.solution.y,
            info=self._solver.info
        )
