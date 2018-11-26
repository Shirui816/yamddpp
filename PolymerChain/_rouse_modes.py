from numba import guvectorize
from numba import float64
import numpy as np


@guvectorize([(float64[:, :], float64[:, :], float64[:, :])], '(n,m),(m,k)->(n,k)',
             target='parallel')  # target='cpu','gpu'
def normal_mode_dot(a, b, ret):  # much more faster than np.tensordot or np.eisum
    r"""Vectorized universal function.
    :param a: np.ndarray, factors with (n_modes, chain_length)
    :param b: np.ndarray, positions with (..., chain_length, n_dimensions),
    axes will be assigned automatically to last 2 axes due to the signatures.
    :param ret: np.ndarray, results. (..., n_modes, n_dimensions)
    :return: np.ndarray ret.
    """
    for i in range(a.shape[0]):
        for k in range(b.shape[1]):
            tmp = 0.
            for j in range(a.shape[1]):
                tmp += a[i, j] * b[j, k]
            ret[i, k] = tmp


def normal_modes(pos, modes=False):
    r"""Normal modes of polymer chains (same chain lengths).
    :param pos: np.ndarray, positions in (n_frames (optional), n_chains (optional), chain_length, n_dimensions)
    :param modes: iterable, modes to calculate
    :return: np.ndarray, normal modes (..., n_modes, n_dimensions)
    """
    chain_length = pos.shape[-2]
    # given modes or all 1 - n modes by default.
    modes = np.asarray(modes) - 1 / 2 if modes is not False else \
        np.arange(1, chain_length + 1)
    factors = 1 / chain_length * np.asarray(
        [np.cos(p * np.pi / chain_length * (np.arange(1, chain_length + 1) - 1 / 2)) for p in modes]
    )
    return normal_mode_dot(factors, pos)
