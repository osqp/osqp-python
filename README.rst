Python interface for OSQP
=========================

.. image:: https://travis-ci.org/oxfordcontrol/osqp-python.svg?branch=master
    :target: https://travis-ci.org/oxfordcontrol/osqp-python

.. image:: https://ci.appveyor.com/api/projects/status/79kru1jue00xsr2l/branch/master?svg=true
    :target: https://ci.appveyor.com/project/bstellato/osqp-python/


Python wrapper for `OSQP <http://osqp.readthedocs.io/>`__: the Operator
Splitting QP Solver.

The OSQP (Operator Splitting Quadratic Program) solver is a numerical
optimization package for solving problems in the form

::

    minimize        0.5 x' P x + q' x

    subject to      l <= A x <= u

where ``x in R^n`` is the optimization variable. The objective function
is defined by a positive semidefinite matrix ``P in S^n_+`` and vector
``q in R^n``. The linear constraints are defined by matrix
``A in R^{m x n}`` and vectors ``l in R^m U {-inf}^m``,
``u in R^m U {+inf}^m``.

Documentation
-------------

The interface is documented `here <https://osqp.org/>`__.


Packaging
---------
This repository just performs the tests and does not build any python package.
We use the external repositories for `conda recipes <https://github.com/oxfordcontrol/osqp-recipes>`_ and `python wheels <https://github.com/oxfordcontrol/osqp-wheels>`_.
