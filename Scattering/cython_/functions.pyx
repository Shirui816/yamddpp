from libc.math cimport sqrt, floor
cimport cython
cimport numpy as np
import numpy as np

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def real_input(np.ndarray[double, ndim=3] m_xyz, np.ndarray[double, ndim=2] r, double r_max, double r_bin):
    cdef long i, j, k
    cdef long n = int(floor(r_max / r_bin))
    cdef np.ndarray[double, ndim=1] m_r = np.zeros((n,))
    cdef np.ndarray[double, ndim=1] rs = np.zeros((n,))
    cdef np.ndarray[long, ndim=1] ct = np.zeros((n,), dtype=np.int64)
    for i in range(0, m_xyz.shape[0]):
        x = r[0, i]
        for j in range(0, m_xyz.shape[1]):
            y = r[1, j]
            for k in range(0, m_xyz.shape[2]):
                z = r[2, k]
                r_ = sqrt(x * x + y * y + z * z)
                if r_ < r_max:
                    idx = floor(r_ / r_bin)
                    m_r[idx] += m_xyz[i, j, k]
                    rs[idx] += r_
                    ct[idx] += 1
    ct[ct == 0] = 1
    return rs / ct, m_r / ct


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def complex_input(np.ndarray[double complex, ndim=3] m_xyz, np.ndarray[double, ndim=2] r,
                       double r_max, double r_bin):
    cdef long i, j, k
    cdef long n = int(floor(r_max / r_bin))
    cdef np.ndarray[double complex, ndim=1] m_r = np.zeros((n,), dtype=np.complex128)
    cdef np.ndarray[double, ndim=1] rs = np.zeros((n,))
    cdef np.ndarray[long, ndim=1] ct = np.zeros((n,), dtype=np.int64)
    for i in range(0, m_xyz.shape[0]):
        x = r[0, i]
        for j in range(0, m_xyz.shape[1]):
            y = r[1, j]
            for k in range(0, m_xyz.shape[2]):
                z = r[2, k]
                r_ = sqrt(x * x + y * y + z * z)
                if r_ < r_max:
                    idx = floor(r_ / r_bin)
                    m_r[idx] += m_xyz[i, j, k]
                    rs[idx] += r_
                    ct[idx] += 1
    ct[ct == 0] = 1
    return rs / ct, m_r / ct
