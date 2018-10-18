import numpy as np
from scipy.stats import circmean

_bins = 50000
_bin_size = 2 * np.pi / _bins
_d = np.linspace(0, 2*np.pi, _bins, endpoint=False)


def _circfuncs_common(samples, high, low):
    r"""
    :param samples: np.ndarray
    :param high: np.ndarray
    :param low: np.ndarray
    :return: np.ndarray
    """
    samples = np.asarray(samples)
    if samples.size == 0:
        return np.nan, np.nan
    return (samples - low)*2.*np.pi / (high - low)


def _circ_mean(ang, axis=None):
    r"""Circular mean of angles in (0, 2\pi].
    :param ang: np.ndarray, angular values in (0, 2\pi]
    :param axis: int, about which axis
    :return: np.ndarray
    """
    s = np.sin(ang).sum(axis=axis)
    c = np.cos(ang).sum(axis=axis)
    res = np.arctan2(s, c)
    return res % (2 * np.pi)


def _circ_midpoint(x, _bin_size, _bins):
    r"""Deduplicate using histogram then calculate circular median.
    :param x: np.ndarray.
    :param _bin_size: float, bin size.
    :param _bins: int, how many bins.
    :return: np.ndarray
    """
    p = np.bincount((x/_bin_size).astype(np.int), minlength=_bins)
    return _circ_mean(_d[p > 0])


def com(x, hi, lo, axis=None, median=False):
    r"""Compute COM or median of PBC datas.
    :param x: np.ndarray, input coordinates.
    :param hi: np.ndarry, box.high
    :param lo: np.ndarray, box.low
    :param axis: int, mean along give axis.
    :param median: bool, calculate median or COM
    :return: np.ndarray, median or COM
    """
    box = hi - lo
    x_ang = _circfuncs_common(x, hi, lo)
    if not median:
        ang = _circ_mean(x_ang, axis=axis)
        return ang / 2 / np.pi * box + lo
    mean_ang = np.apply_along_axis(_circ_midpoint, axis,
                                   x_ang, _bin_size, _bins)
    return mean_ang / 2 / np.pi * box + lo


# Abandoned:


def com_rho(x, hi, lo, n=10000, median=False):
    r"""
    :param x: 2d np.ndarray with (n_samples, ndim), 1 object
    only
    :param hi: 1d np.ndarray with (ndim,)
    :param lo: 1d np.ndarray with (ndim,)
    :param n: bins to be histogramed
    :param median: com or median
    :return: com or median
    """
    assert x.ndim == 2
    ret = np.zeros(x.shape[1])
    box = hi - lo
    w = np.exp(-2j * np.pi * np.arange(n) / n)
    for d, b in enumerate(box):
        y, _ = np.histogram(x.T[d], bins=n, range=(lo[d], hi[d]))
        if median:
            y = (y > 0).astype(np.float)
        a = np.angle((w.dot(y)).conj()) % (2 * np.pi)
        ret[d] = lo[d] + box[d] * a / 2 / np.pi
    return ret


def com_abandoned(x, hi, lo, density=False, n=10000, median=False):
    r"""
    :param x: 2d np.ndarray with (n_samples, ndim) or 3d np.ndarray
    with (n_objects, n_samples, ndim)
    :param hi: 1d np.ndarray with (ndim,)
    :param lo: 1d np.ndarray with (ndim,)
    :param axis: axis == 0 for 2d (1 object) or axis == 1 for 3d (multiple
    objects)
    :param density: using density mode
    :param n: numbers of bins to be histogramed
    :param median: com or median
    :return: com(s) or median(s)
    """
    assert x.ndim == 3 or x.ndim == 2
    if (not density) and median:
        raise ValueError("Sorry, currently only density mode supports median!")
    if not density:
        axis = 0 if x.ndim == 2 else 1
        return circmean(x, high=hi, low=lo, axis=axis)
    if not isinstance(hi, np.ndarray):
        hi = np.array([hi] * x.shape[-1])
    if not isinstance(lo, np.ndarray):
        lo = np.array([lo] * x.shape[-1])
    if x.ndim == 3:
        return np.array([com_rho(_, hi, lo, n, median)
                        for _ in x])
    elif x.ndim == 2:
        return com_rho(x, hi, lo, n, median)
