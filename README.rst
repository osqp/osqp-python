Python interface for OSQP: The Operator Splitting QP Solver 
===========================================================

|Build Status| |Build status| |codecov.io|

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

The interface is documented `here <http://osqp.readthedocs.io/>`__.

.. |Build Status| image:: https://travis-ci.org/oxfordcontrol/OSQP.jl.svg?branch=master
   :target: https://travis-ci.org/oxfordcontrol/OSQP.jl
.. |Build status| image:: https://ci.appveyor.com/api/projects/status/t9t79ncgl9449k1a/branch/master?svg=true
   :target: https://ci.appveyor.com/project/bstellato/osqp-jl-v2y0g/branch/master
.. |codecov.io| image:: http://codecov.io/github/oxfordcontrol/OSQP.jl/coverage.svg?branch=master
   :target: http://codecov.io/github/oxfordcontrol/OSQP.jl?branch=master
