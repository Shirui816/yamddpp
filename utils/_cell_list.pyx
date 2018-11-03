from libc.math cimport sqrt, floor
cimport cython
cimport numpy as np
import numpy as np

# Following 2 funcs need to extend to satisfy cases of any dimensions.
@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cell_id(np.ndarray[ndim=1, double] x, np.ndarray[ndim=1, double] box, np.ndarray[ndim=1, long] ibox):
    cdef np.ndarray[ndim=1, long] ind = np.asarray(((x / box + 0.5) * ibox), dtype=np.int64)
    return ind[0] + ind[1] * ibox[0] + ind[2] * ibox[1] * ibox[0]

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def i_cell(np.ndarray[ndim=1, long] cid, np.ndarray[ndim=1, long] ibox):
    cdef np.ndarray[ndim=1, long] ind = np.asarray((cid + ibox) % ibox, dtype=np.int64)
    return ind[0] + ind[1] * ibox[0] + ind[2] * ibox[1] * ibox[0]

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cell_neighbours(np.ndarray[ndim=1, long] ic, np.ndarray[ndim=1, long] ibox):
    cdef long n_dim = ibox.shape[0]
    cdef long ct = 0
    cdef np.ndarray[ndim=1, long] ret = np.zeros(n_dim ** 3, )
    for ind in np.ndindex((3,) * n_dim):
        ind = np.asarray(ind) - 1
        ret[ct] = i_cell(ind + ic, ibox)
        ct += 1
    return ret

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def box_map(np.ndarray[ndim=1, double] box, double r_cut):
    cdef np.ndarray[ndim=1, long] ibox = np.asarray(box / r_cut, dtype=np.int64)
    cdef long ic
    cdef long n_dim = box.shape[0]
    cdef np.ndarray[ndim=2, long] ret = np.zeros((np.multiply.reduce(ibox), n_dim ** 3))
    for ind in np.ndindex(ibox):
        ind = np.asarray(ind)
        ic = i_cell(ind, ibox)
        ret[ic] = cell_neighbours(ind, ibox)
    return ret, ibox

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def linked_cl(np.ndarray[ndim=2, double] pos, np.ndarray[ndim=1, double] box,
              np.ndarray[ndim=1, long] ibox, long n_jobs=4):
    cdef np.ndarray[ndim=1, long] head = np.zeros(np.multiply.reduce(ibox), dtype=np.int64) - 1
    cdef np.ndarray[ndim=1, long] body = np.zeros(pos.shape[0], dtype=np.int64)
    cdef long i
    for i in range(pos.shape[0]):
        ic = cell_id(pos[i], box, ibox)
        body[i] = head[ic]
        head[ic] = i
    return head, body
