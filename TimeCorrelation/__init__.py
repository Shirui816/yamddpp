from ._mat_ac import vec_ac
from ._mat_ac import mat_ac
import numpy as np


def next_regular(target):
    r"""Copied from scipy.signal.fftconvolve
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

            n = p2 * p35
            if n == target:
                return n
            elif n < match:
                match = n
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
    Usually used in time-correlation, one axis is enough. Don't
    use this function to calculate autocorrelation, for
    cross_correlate(in1, in1[::-1]) is on interval [-len(in1), len(in1)),
    not on [0, len(in1)), in this case, output[len(in1)-1:] is the correct
    output for in1 == in2 or in1.size == in2.size, due to `np.flip` is
    applied on in2 here. Or generally,
    ifft(fft(a, n=na+nb-1) * fft(b, n=na+nb-1).conj())[:na] ==
    np.correlate(a, b, 'full')[nb-1:] for na >= nb... See `same' and `valid' mode
    in `scipy.fft.convolve' (`_centered' function).

    >>> a = np.random.random(10)
    >>> b = np.random.random(10)
    >>> np.allclose(ifft(fft(a,19)*fft(b,19).conj())[:10], np.correlate(a,b,'full')[9:])
    True

    :param in1: np.ndarray
    :param in2: np.ndarray
    :param axis: int
    :return: np.ndarray, `full' mode of np.correlate result.
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
    f = next_regular(s)
    return ifft(fft(in1, axis=axis, n=f) *
                fft(np.flip(in2, axis=axis).conj(), axis=axis, n=f),
                axis=axis, n=f)[:s]
