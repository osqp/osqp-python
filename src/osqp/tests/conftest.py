import os
from osqp import algebra_available


def pytest_generate_tests(metafunc):

    # detect env vars to decide which algebras to include/skip
    algebras_include = os.environ.get('OSQP_TEST_ALGEBRA_INCLUDE', 'builtin mkl-direct mkl-indirect cuda').split()
    algebras_skip = os.environ.get('OSQP_TEST_ALGEBRA_SKIP', '').split()
    algebras = [x for x in algebras_include if x not in algebras_skip]

    parameters = ('algebra', 'solver_type', 'atol', 'rtol', 'decimal_tol')
    values = []
    if algebra_available('builtin') and 'builtin' in algebras:
        values.append(
            ('builtin', 'direct', 1e-3, 1e-4, 4),
        )
    if algebra_available('mkl') and 'mkl-direct' in algebras:
        values.append(
            ('mkl', 'direct', 1e-3, 1e-4, 4),
        )
    if algebra_available('mkl') and 'mkl-indirect' in algebras:
        values.append(
            ('mkl', 'indirect', 1e-3, 1e-4, 3),
        )
    if algebra_available('cuda') and 'cuda' in algebras:
        values.append(
            ('cuda', 'indirect', 1e-2, 1e-3, 2),
        )

    metafunc.parametrize(parameters, values)
