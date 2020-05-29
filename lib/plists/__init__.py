from numba import cuda, int64, void

from .nlist import nlist


@cuda.jit(int64(int64[:], int64[:]), device=True)
def cu_ravel_index_f_pbc(i, dim):  # ravel index in Fortran way.
    ret = (i[0] + dim[0]) % dim[0]
    tmp = dim[0]
    for k in range(1, dim.shape[0]):
        ret += ((i[k] + dim[k]) % dim[k]) * tmp
        tmp *= dim[k]
    return ret


@cuda.jit(void(int64, int64[:], int64[:]), device=True)
def cu_unravel_index_f(i, dim, ret):  # unravel index in Fortran way.
    for k in range(dim.shape[0]):
        ret[k] = int(i % dim[k])
        i = (i - ret[k]) / dim[k]
