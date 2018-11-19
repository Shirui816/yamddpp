from ._hist_by_mod import hist_vec_by_r
from ._hist_by_mod_cu import hist_vec_by_r_cu
import numpy as np


def rfft2fft(rfft, n):
    r"""Extend rfft output to FFT outputs.

    See numpy.fft.rfftn for details, the last axis of output of rfftn is always
    n // 2 + 1.

    Example:

    >>> a = np.random.random((10,20,30))
    >>> np.allclose(rfft2fft(np.fft.rfftn(a), 30), np.fft.fftn(a))
    True

    :param rfft: np.ndarray
    :param n: int, last axis of desired fft results
    :return: np.ndarray, fft results.
    :raises: ValueError if n < (rfft.shape[-1] - 1) * 2
    """
    if n < (rfft.shape[-1] - 1) * 2:
        raise ValueError("Shape of FFT cannot smaller than RFFT outputs!")
    n_dim = rfft.ndim
    fslice = tuple([slice(0, _) for _ in rfft.shape])
    lslice = np.arange(n - n // 2 - 1, 0, -1)
    pad_axes = [(0, 1)] * (n_dim - 1) + [(0, 0)]
    flip_axes = tuple(range(n_dim - 1))
    # fftn(a) = np.concatenate([rfftn(a),
    # conj(rfftn(a))[-np.arange(i),-np.arange(j)...,np.arange(k-k//2-1,0,-1)]], axis=-1)
    return np.concatenate([rfft, np.flip(np.pad(rfft.conj(), pad_axes, 'wrap'),
                                         axis=flip_axes)[fslice][..., lslice]], axis=-1)
