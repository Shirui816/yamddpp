import numpy as np


def vec_ac(x, cum=True):
    r"""Vector autocorrelation function with samples.
    :param x: np.ndarray -> (n_frames, n_vectors, ...)
    :param cum: bool, take average of n_vectors or not.
    :return: np.ndarray -> (n_frames,) of vector autocorrelation
            or (n_frames, n_vectors) if `cum=False'.
    """
    n = x.shape[0]
    summing_axes = tuple(range(1, x.ndim)) if cum else \
        tuple(range(2, x.ndim))
    norm = np.arange(n, 0, -1)
    if not cum:
        norm = np.expand_dims(norm, axis=-1)
    # summing over samples and dimension or just dimension
    return np.fft.irfft(np.sum(abs(np.fft.rfft(x, axis=0, n=n * 2)) ** 2,
                               axis=summing_axes))[:n].real / norm


def mat_ac(x):
    r"""Matrix autocorrelation function.
    :param x: np.ndarray -> (n_frames, ...) of input
    :return: np.ndarray -> (n_frames, ...) of output
    """
    n = x.shape[0]
    norm = np.arange(n, 0, -1).reshape(n, *[1] * (x.ndim - 1))
    return np.fft.irfft(abs(np.fft.rfft(x, axis=0, n=n * 2)) ** 2,
                        axis=0)[:n].real / norm


def mat_ac_comp(x):
    r"""Matrix autocorrelation function.
    :param x: np.ndarray -> (n_frames, ...) of input
    :return: np.ndarray -> (n_frames, ...) of output
    """
    n = x.shape[0]
    norm = np.arange(n, 0, -1).reshape(n, *[1] * (x.ndim - 1))
    return np.fft.ifft(abs(np.fft.fft(x, axis=0, n=n * 2)) ** 2,
                       axis=0)[:n] / norm
