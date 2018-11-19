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
    n = next_regular(2 * x.shape[0])
    summing_axes = tuple(range(1, x.ndim)) if cum else \
        tuple(range(2, x.ndim))
    norm = np.arange(x.shape[0], 0, -1)
    if not cum:
        norm = np.expand_dims(norm, axis=-1)
    # summing over samples and dimension or just dimension
    return ifft(np.sum(abs(fft(x, axis=0, n=n)) ** 2,
                       axis=summing_axes), axis=0, n=n)[:x.shape[0]].real / norm


def mat_ac(x):
    r"""Matrix autocorrelation function.
    :param x: np.ndarray -> (n_frames, ...) of input
    :return: np.ndarray -> (n_frames, ...) of output
    """
    fft = np.fft.rfft
    ifft = np.fft.irfft
    if np.issubdtype(x.dtype, np.complex):
        fft = np.fft.fft
        ifft = np.fft.ifft
    n = next_regular(2 * x.shape[0])
    norm = np.arange(x.shape[0], 0, -1).reshape(x.shape[0], *[1] * (x.ndim - 1))
    return ifft(abs(fft(x, axis=0, n=n)) ** 2,
                axis=0, n=n)[:x.shape[0]].real / norm
