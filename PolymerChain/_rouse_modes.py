from numba import guvectorize
from numba import float64
import numpy as np


@guvectorize([(float64[:, :], float64[:, :], float64[:, :])], '(n,p),(p,m)->(n,m)',
             target='parallel')  # target='cpu','gpu'
def _batch_dot(a, b, ret):  # much more faster than np.tensordot or np.einsum
    r"""Vectorized universal function.
    :param a: np.ndarray, factors with (n_modes, chain_length)
    :param b: np.ndarray, positions with (..., chain_length, n_dimensions),
    axes will be assigned automatically to last 2 axes due to the signatures.
    :param ret: np.ndarray, results. (..., n_modes, n_dimensions)
    :return: np.ndarray ret.
    """
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            tmp = 0.
            for k in range(a.shape[1]):
                tmp += a[i, k] * b[k, j]
            ret[i, j] = tmp


def normal_modes(pos, modes=None):
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
    >>>x = np.random.random((20, 10, 250, 3))  # 20 frames, 10 chains, 250 beads in 3 dimensions for example.
    >>>np.allclose(np.asarray([np.swapaxes(np.tensordot(factors, x[i], axes=[1, 1]), 0, 1) for i in range(20)]),
    normal_modes(x))
    >>>True

    Speed test:

    In [1]: pos = np.random.random((1000,1000,100,3))

    In [2]: %timeit normal_modes(pos)
    1.42 s ± 96.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    In [3]: chain_length = 100

    In [4]: modes = np.arange(1,101)

    In [5]: factors = 1 / chain_length * np.asarray(
    ...:         [np.cos(p * np.pi / chain_length * (np.arange(1, chain_length + 1))) for p in modes]
    ...:     )

    In [6]: %timeit np.asarray([np.swapaxes(np.tensordot(factors, pos[i], axes=[1, 1]), 0, 1)
    ....:                       for i in range(pos.shape[0])])
    9.07 s ± 1.14 s per loop (mean ± std. dev. of 7 runs, 1 loop each)

    Or by np.einsum, which is convenient but slow:

    In [1]: b = np.random.random((30,100,250,3))

    In [2]: np.allclose(normal_modes(b),np.einsum('ij,abjk->abik', factors, b))
    Out[2]: True

    In [3]: %timeit np.einsum('ab,cdbe->cdae', factors, b)
    3.5 s ± 1.54 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    In [4]: %timeit normal_modes(b)
    324 ms ± 1.74 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    :param pos: np.ndarray, positions in (n_frames (optional), n_chains (optional), chain_length, n_dimensions)
    :param modes: iterable, modes to calculate. mode 1 ~ chain_length are calculated by default.
    :return: np.ndarray, normal modes (..., n_modes, n_dimensions)
    """
    chain_length = pos.shape[-2]
    # given modes or all 1 ~ n modes by default.
    modes = np.asarray(modes) - 1 / 2 if modes is not None else \
        np.arange(1, chain_length + 1)
    # def was taken from Iwao Teraoka, Polymer Solutions, pp. 223
    factors = 1 / chain_length * np.asarray(
        [np.cos(p * np.pi / chain_length * (np.arange(1, chain_length + 1))) for p in modes]
    )
    return _batch_dot(factors, pos)
