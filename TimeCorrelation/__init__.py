from ._mat_ac import vec_ac
from ._mat_ac import mat_ac
import numpy as np


def _next_regular(target):
    r"""Copy from scipy.signal.fftconvolve
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.
    Target must be a positive integer.
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target - 1)):
        return target

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            # Quickly find next power of 2 >= quotient
            try:
                p2 = 2 ** ((quotient - 1).bit_length())
            except AttributeError:
                # Fallback for Python <2.7
                p2 = 2 ** (len(bin(quotient - 1)) - 2)

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match


def cross_correlate(in1, in2, axis=0):
    r"""Cross-correlation of matrices along given axis.
    Usually used in time-correlation, axis is enough.
    :param in1: np.ndarray
    :param in2: np.ndarray
    :param axis: int
    :return: `full' mode of np.correlate result.
    """
    fft = np.fft.rfft
    ifft = np.fft.irfft
    if (np.issubdtype(in1.dtype, np.complex) or
            np.issubdtype(in2.dtype, np.complex)):
        fft = np.fft.fft
        ifft = np.fft.ifft
    s1 = in1.shape[axis]
    s2 = in2.shape[axis]
    s = s1 + s2 - 1
    f = _next_regular(s)
    return ifft(fft(in1, axis=axis, n=f) *
                fft(np.flip(in2, axis=axis).conj(), axis=axis, n=f),
                axis=axis, n=f)[:s]
