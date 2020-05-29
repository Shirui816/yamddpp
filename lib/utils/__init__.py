from numba import cuda, int64, void

from ._cell_list import cell_id
from ._cell_list import get_from_cell
from ._cell_list import linked_cl
from ._cell_list import unravel_index_f
from ._cell_list_cu import cu_cell_id
from ._cell_list_cu import cu_cell_list_argsort, cu_cell_ind, cu_cell_count
from ._hist_by_mod import hist_vec_by_r
from ._hist_by_mod_cu import hist_vec_by_r_cu
from ._nlist_cu import cu_nl
from ._nlist_cu import cu_nl_strain
from ._utils import cu_max_int, cu_set_to_int, cu_mat_dot_v_pbc_dist, cu_mat_dot_v, cu_v_mod, cu_mat_dot_v_pbc
from ._utils import pbc_dist_cu
from ._utils import ravel_index_f_cu, unravel_index_f_cu, add_local_arr_mois_1
from ._utils import rfft2fft


@cuda.jit("void(int64[:], int64)")
def cu_set_to_int(arr, val):
    i = cuda.grid(1)
    if i >= arr.shape[0]:
        return
    arr[i] = val


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
