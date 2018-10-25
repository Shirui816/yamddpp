import numpy as np


def vec_ac(x, cum=True):
    r"""Vector autocorrelation function with samples.
    :param x: np.ndarray -> (n_frames, n_vectors, ...)
    :param cum: bool, take average of n_vectors or not.
    :return: np.ndarray -> (n_frames,) of vector autocorrelation
            or (n_frames, n_vectors) if `cum=False'.
    """
    n = x.shape[0]
    ft_x = np.fft.rfft(x, axis=0, n=n * 2)  # FFT over time axis
    summing_axes = tuple(range(1, x.ndim)) if cum else \
        tuple(range(2, x.ndim))
    ft_corr = np.sum(abs(ft_x) ** 2, axis=summing_axes)
    # summing over samples and dimension or just dimension
    return np.fft.irfft(ft_corr)[:n].real / np.arange(n, 0, -1)


def mat_ac(x):
    r"""Matrix autocorrelation function.
    :param x: np.ndarray -> (n_frames, ...) of input
    :return: np.ndarray -> (n_frames, ...) of output
    """
    n = x.shape[0]
    ft_x = np.fft.rfft(x, axis=0, n=n * 2)  # FFT over time axis
    ft_corr = abs(ft_x) ** 2
    return np.fft.irfft(ft_corr)[:n].real / np.arange(n, 0, -1)
