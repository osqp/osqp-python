import osqp.ext as ext
from osqp.ext import Hello
from osqp.ext import OSQPSettings


def test_greet():
    h = Hello()
    assert h.greet() == 'Hello Pybind11'


def test_osqp_rho():
    settings = OSQPSettings()
    assert settings.rho == 0.1