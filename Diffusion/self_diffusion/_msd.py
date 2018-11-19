import numpy as np
from Correlation import vec_ac


def msd_square(x, cum=True):
    r"""
    :param x: np.ndarray, all vectors (n_frames, n_particles, n_dimension)
    :param cum: bool, summing n_particles or not
    :return: np.ndarray, msd values
    """
    n, n_samples = x.shape[0], x.shape[1]
    summing_axes = (1, 2) if cum else 2
    xt = np.square(x).sum(axis=summing_axes)
    x0 = 2 * xt.sum(axis=0)
    _shape = (n,) if cum else (n, n_samples)
    xm = np.zeros(_shape)
    xm[0] = x0 / n
    for m in range(1, n):
        x0 = x0 - xt[m - 1] - xt[n - m]
        xm[m] = x0 / (n - m)
    return xm


def msd(x, cum=True):
    r"""
    :param x: np.ndarray, (n_frames, n_particles, n_dim)
    :param cum: bool, summing n_particles or not.
    :return: np.ndarray, msd
    """
    return msd_square(x, cum=cum) - 2 * vec_ac(x, cum=cum)
