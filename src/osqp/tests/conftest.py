from osqp import algebra_available


def pytest_generate_tests(metafunc):
    parameters = ('algebra', 'solver_type', 'atol', 'rtol', 'decimal_tol')
    values = []
    if algebra_available('builtin'):
        values.extend(
            [
                ('builtin', 'direct', 1e-3, 1e-4, 4),
            ]
        )
    if algebra_available('mkl'):
        values.extend(
            [
                ('mkl', 'direct', 1e-3, 1e-4, 4),
                ('mkl', 'indirect', 1e-3, 1e-4, 3),
            ]
        )
    if algebra_available('cuda'):
        values.extend(
            [
                ('cuda', 'indirect', 1e-2, 1e-3, 2),
            ]
        )

    metafunc.parametrize(parameters, values)
