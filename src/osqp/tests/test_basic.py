import osqp.ext as ext
from osqp.ext import Hello


def test_greet():
    h = Hello()
    assert h.greet() == 'Hello Pybind11'


def test_add():
    assert ext.add(3, 6) == 9


def test_subtract():
    assert ext.subtract(3, 6) == -3
