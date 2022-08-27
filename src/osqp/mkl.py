from osqp.interface import OSQP as _OSQP


class OSQP(_OSQP):
    def __init__(self, *args, **kwargs):
        super(OSQP, self).__init__(*args, **kwargs, algebra='mkl')
