import os.path
import numpy as np


def load_high_accuracy(test_name):
    npz = os.path.join(os.path.dirname(__file__), 'solutions', f'{test_name}.npz')
    npzfile = np.load(npz)
    return npzfile['x_val'], npzfile['y_val'], npzfile['obj']
