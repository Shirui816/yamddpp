import numpy as np
from . import next_regular


def vec_ac(x, cum=True):
    r"""Vector autocorrelation function with samples.
    :param x: np.ndarray -> (n_frames, n_vectors, ...)
    :param cum: bool, take average of n_vectors or not.
    :return: np.ndarray -> (n_frames,) of vector autocorrelation
            or (n_frames, n_vectors) if `cum=False'.
    """
    fft = np.fft.rfft
    ifft = np.fft.irfft
    if np.issubdtype(x.dtype, np.complex):
        fft = np.fft.fft
        ifft = np.fft.ifft
    n = x.shape[0]
    s = next_regular(2 * n)
    summing_axes = tuple(range(1, x.ndim)) if cum else \
        tuple(range(2, x.ndim))
    norm = np.arange(n, 0, -1)
    if not cum:
        norm = np.expand_dims(norm, axis=-1)
    # summing over samples and dimension or just dimension
    return ifft(np.sum(abs(fft(x, axis=0, n=s)) ** 2,
                       axis=summing_axes), axis=0, n=s)[:n].real / norm


def mat_ac(x, axes=None):
    r"""Matrix autocorrelation function.
    :param x: np.ndarray -> (n_frames, ...) of input
    :param axes: tuple or int, axes that summing up.
    :return: np.ndarray -> (n_frames, ...) of output
    :raises: ValueError, if axes contains 0.
    """
    fft = np.fft.rfft
    ifft = np.fft.irfft
    if np.issubdtype(x.dtype, np.complex):
        fft = np.fft.fft
        ifft = np.fft.ifft
    n = x.shape[0]
    s = next_regular(2 * n)  # 2 * n - 1 is fine.
    norm = np.arange(n, 0, -1).reshape(n, *[1] * (x.ndim - 1))
    if axes is None:
        return ifft(abs(fft(x, axis=0, n=s)) ** 2,
                    axis=0, n=s)[:n].real / norm
    else:
        axes = np.atleast_1d(np.asarray(axes, dtype=np.int))
        if 0 in axes:
            raise ValueError("The 1st axis should be time axis!")
        norm = norm.reshape(n, *[1] * (x.ndim - 1 - axes.size))
        return ifft(np.sum(abs(fft(x, axis=0, n=s)) ** 2, axis=tuple(axes)),
                    axis=0, n=s)[:n].real / norm
