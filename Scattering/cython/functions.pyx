from libc.math cimport sqrt, floor
cimport cython
cimport numpy as np

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def to_sq(np.ndarray[double, ndim=3] s_q, np.ndarray[double, ndim=2] freq, double q_max, double q_bin):
    cdef long i, j, k, n = int(floor(q_max / q_bin))
    cdef np.ndarray[double, ndim=1] sq = np.zeros((n,))
    cdef np.ndarray[double, ndim=1] qs = np.zeros((n,))
    cdef np.ndarray[long, ndim=1] ct = np.zeros((n,))
    for i in range(1, s_q.shape[0]):
        qx = freq[0, i]
        for j in range(1, s_q.shape[1]):
            qy = freq[1, j]
            for k in range(1, s_q.shape[2]):  # Zero-freq is meaningless
                qz = freq[2, k]
                q = sqrt(qx * qx + qy * qy + qz * qz)
                if q < q_max:
                    idx = floor(q / q_bin)
                    sq[idx] += s_q[i, j, k]
                    qs[idx] += q
                    ct[idx] += 1
    ct[ct == 0] = 1
    return qs / ct, sq / ct
