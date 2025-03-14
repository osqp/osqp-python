[![PyPI version](https://badge.fury.io/py/osqp.svg)](https://badge.fury.io/py/osqp)
[![Python 3.8‒3.13](https://img.shields.io/badge/python-3.8%E2%80%923.13-blue)](https://www.python.org)
[![Build](https://github.com/osqp/osqp-python/actions/workflows/build_default.yml/badge.svg)](https://github.com/osqp/osqp-python/actions/workflows/build_default.yml)

# OSQP Python
Python wrapper for [OSQP](https://osqp.org): The Operator Splitting QP solver.

The OSQP (Operator Splitting Quadratic Program) solver is a numerical
optimization package for solving problems in the form

$$\begin{array}{ll}
    \mbox{minimize} & \frac{1}{2} x^T P x + q^T x \\
    \mbox{subject to} & l \le A x \le u
\end{array}
$$

where $\( x \in \mathbf{R}^n \)$ is the optimization variable and $\( P \in \mathbf{S}^{n}_{+} \)$ is a positive semidefinite matrix.

## Installation
To install `osqp` for python, make sure that you're using a recent version of `pip` (`pip install --upgrade pip`)
and then use ``pip install osqp``.

To install `osqp` from source, clone the repository (`git clone https://github.com/osqp/osqp-python`)
and run `pip install .` from inside the cloned folder.

## Documentation
The interface is documented [here](https://osqp.org/docs/interfaces/python.html).
