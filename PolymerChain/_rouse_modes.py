from numba import guvectorize
from numba import float64
import numpy as np


@guvectorize([(float64[:, :], float64[:, :], float64[:, :])], '(n,m),(m,k)->(n,k)',
             target='parallel')  # target='cpu','gpu'
def _normal_mode_dot(a, b, ret):  # much more faster than np.tensordot or np.eisum
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
    r"""Normal modes of polymer chains (same chain lengths), definition of $q_i$ was
    taken from Iwao Teraoka, Polymer Solutions, pp. 223.

    >>>Xp = lambda x, p : np.mean(np.cos(p * np.pi/x.shape[1] * np.arange(1,x.shape[1]+1))[:,None] * x[0], axis=0)
    >>>x = np.random.random((1, 250, 3))
    >>>np.allclose(normal_modes(x), np.asarray([Xp(x, p) for p in range(1, 251)]))
    >>>True
    >>>factors = 1 / chain_length * np.asarray(
        [np.cos(p * np.pi / chain_length * (np.arange(1, chain_length + 1))) for p in modes]
    )
    >>>np.allclose(normal_modes(x), np.swapaxes(np.tensordot(factors, x, axes=[1, 1]), 0, 1))
    >>>True
    >>>x = np.random.random((20, 10, 250, 3))  # 20 frames, 10 chains, 250 beads in 3 dimension for example.
    >>>np.allclose(np.asarray([np.swapaxes(np.tensordot(factors, x[i], axes=[1, 1]), 0, 1))
                               for i in range(20)]), normal_mode(x))
    >>>True

    :param pos: np.ndarray, positions in (n_frames (optional), n_chains (optional), chain_length, n_dimensions)
    :param modes: iterable, modes to calculate. mode 1 ~ chain_length are calculated by default.
    :return: np.ndarray, normal modes (..., n_modes, n_dimensions)
    """
    chain_length = pos.shape[-2]
    # given modes or all 1 ~ n modes by default.
    modes = np.asarray(modes) - 1 / 2 if modes is not False else \
        np.arange(1, chain_length + 1)
    # def was taken from Iwao Teraoka, Polymer Solutions, pp. 223
    factors = 1 / chain_length * np.asarray(
        [np.cos(p * np.pi / chain_length * (np.arange(1, chain_length + 1))) for p in modes]
    )
    return _normal_mode_dot(factors, pos)
