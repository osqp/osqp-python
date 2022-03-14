import os.path
import numpy as np


rel_tol = 1e-03
abs_tol = 1e-04
decimal_tol = 4


def load_high_accuracy(test_name):
    npz = os.path.join(os.path.dirname(__file__), 'solutions', f'{test_name}.npz')
    npzfile = np.load(npz)
    return npzfile['x_val'], npzfile['y_val'], npzfile['obj']


# A list of random states, used as a stack
random_states = []


class Random:
    """
    A context manager that pushes a random seed to the stack for reproducible results,
    and pops it on exit.
    """

    def __init__(self, seed=None):
        self.seed = seed

    def __enter__(self):
        if self.seed is not None:
            # Push current state on stack
            random_states.append(np.random.get_state())
            new_state = np.random.RandomState(self.seed)
            np.random.set_state(new_state.get_state())

    def __exit__(self, *args):
        if self.seed is not None:
            np.random.set_state(random_states.pop())
