import numpy as np
from numba import cuda, void, int64, float32, float64

from ..utils import cu_set_to_int
from .clist import clist


@cuda.jit(void(int64[:], int64[:]))
def cu_max_int(arr, arr_max):
    i = cuda.grid(1)
    if i >= arr.shape[0]:
        return
    cuda.atomic.max(arr_max, 0, arr[i])


def _gen_func(dtype):
    from math import floor
    float = float64
    if dtype == np.dtype(np.float32):
        float = float32

    @cuda.jit(float(float[:], float[:], float[:]), device=True)
    def cu_pbc_dist2(a, b, box):
        ret = 0
        for i in range(a.shape[0]):
            d = a[i] - b[i]
            d -= box[i] * floor(d / box[i] + 0.5)
            ret += d ** 2
        return ret

    @cuda.jit(
        void(float[:, :], float[:], float, int64[:, :], int64[:, :], int64[:], int64[:],
             int64[:, :], int64[:], int64[:], int64[:], int64))
    def cu_nlist(x, box, r_cut2, cell_map, cell_list, cell_count, cells, nl, nc, n_max,
                 situation, contain_self):
        pi = cuda.grid(1)
        if pi >= x.shape[0]:
            return
        # xi = cuda.local.array(ndim, dtype=float64)
        # xj = cuda.local.array(ndim, dtype=float64)
        # for l in range(ndim):
        #    xi[l] = x[pi, l]
        ic = cells[pi]
        n_needed = 0
        nn = 0
        xi = x[pi]
        for j in range(cell_map.shape[1]):
            jc = cell_map[ic, j]
            for k in range(cell_count[jc]):
                pj = cell_list[jc, k]
                if contain_self == 0 and pj == pi:  # == 0 means do not contain self.
                    continue
                # for m in range(ndim):
                # xj[m] = x[pj, m]
                r2 = cu_pbc_dist2(xi, x[pj], box)
                if r2 < r_cut2:
                    if nn < nl.shape[1]:
                        nl[pi, nn] = pj
                    else:
                        n_needed = nn + 1
                    nn += 1
        nc[pi] = nn
        if nn > 0:
            cuda.atomic.max(n_max, 0, n_needed)
        if pi == 0:  # reset situation only once while function is called
            situation[0] = 0

    return cu_nlist


class nlist(object):
    def __init__(self, x, box, r_cut, cell_guess=50, n_guess=150, gpu=0, contain_self=0):
        self.x = x
        self.n = x.shape[0]
        self.n_dim = x.shape[1]
        self.box = box
        self.cell_guess = cell_guess
        self.n_guess = n_guess
        self.r_cut2 = r_cut ** 2
        self.r_cut = r_cut
        self.gpu = gpu
        self.tpb = 64
        self.bpg = int(x.shape[0] // self.tpb + 1)
        self.contain_self = contain_self
        # self.situ_zero = np.zeros(1, dtype=np.int64)
        self.update_counts = 0
        self.cu_nlist = _gen_func(x.dtype)
        with cuda.gpus[self.gpu]:
            self.p_n_max = cuda.pinned_array((1,), dtype=np.int64)
            self.p_situation = cuda.pinned_array((1,), dtype=np.int64)
            self.d_n_max = cuda.device_array(1, dtype=np.int64)
            self.d_nl = cuda.device_array((self.n, self.n_guess), dtype=np.int64)
            self.d_nc = cuda.device_array((self.n,), dtype=np.int64)
            self.d_situation = cuda.device_array(1, dtype=np.int64)
        self.clist = clist(self.x, self.box, self.r_cut, cell_guess=self.cell_guess)
        self.d_x = self.clist.d_x
        self.d_box = self.clist.d_box
        self.neighbour_list()

    def neighbour_list(self):
        with cuda.gpus[self.gpu]:
            while True:
                cu_set_to_int[self.bpg, self.tpb](self.d_nc, 0)
                # reset situation while build nlist
                self.cu_nlist[self.bpg, self.tpb](self.d_x,
                                                  self.d_box,
                                                  self.r_cut2,
                                                  self.clist.d_cell_map,
                                                  self.clist.d_cell_list,
                                                  self.clist.d_cell_counts,
                                                  self.clist.d_cells,
                                                  self.d_nl,
                                                  self.d_nc,
                                                  self.d_n_max,
                                                  self.d_situation,
                                                  self.contain_self)
                self.d_n_max.copy_to_host(self.p_n_max)
                cuda.synchronize()
                # n_max = np.array([120])
                if self.d_n_max[0] > self.n_guess:
                    self.n_guess = self.d_n_max[0]
                    self.n_guess = self.n_guess + 8 - (self.n_guess & 7)
                    self.d_nl = cuda.device_array((self.n, self.n_guess), dtype=np.int64)
                else:
                    break

    def update(self):
        self.clist.update()
        self.neighbour_list()
        self.update_counts += 1

    def show(self):
        cell_list = self.clist.d_cell_list.copy_to_host()
        cell_map = self.clist.d_cell_map.copy_to_host()
        cell_counts = self.clist.d_cell_counts.copy_to_host()
        nl = self.d_nl.copy_to_host()
        nc = self.d_nc.copy_to_host()
        cuda.synchronize()
        return cell_list, cell_counts, cell_map, nl, nc
