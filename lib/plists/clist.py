import numpy as np
from numba import cuda
from numba import int64, float32, float64, void
from math import floor

from ..utils import cu_set_to_int
from ..utils import cu_unravel_index_f, cu_ravel_index_f_pbc


def _gen_func(dtype, n_dim):
    float = float64
    if dtype == np.dtype(np.float32):
        float = float32

    @cuda.jit(int64(float[:], float[:], int64[:]), device=True)
    def cu_cell_index(x, box, ibox):
        ret = floor((x[0] / box[0] + 0.5) * ibox[0])
        n_cell = ibox[0]
        for i in range(1, x.shape[0]):
            ret = ret + floor((x[i] / box[i] + 0.5) * ibox[i]) * n_cell
            n_cell = n_cell * ibox[i]
        return ret

    @cuda.jit(void(int64[:], int64[:], int64[:, :]))
    def cu_cell_map(ibox, dim, ret):
        cell_i = cuda.grid(1)
        if cell_i >= ret.shape[0]:
            return
        cell_vec_i = cuda.local.array(n_dim, int64)
        cell_vec_j = cuda.local.array(n_dim, int64)
        cu_unravel_index_f(cell_i, ibox, cell_vec_i)
        for j in range(ret.shape[1]):
            cu_unravel_index_f(j, dim, cell_vec_j)
            for k in range(n_dim):
                cell_vec_j[k] = cell_vec_i[k] + cell_vec_j[k] - 1
            cell_j = cu_ravel_index_f_pbc(cell_vec_j, ibox)
            ret[cell_i, j] = cell_j

    @cuda.jit(void(float[:, :], float[:], int64[:], int64[:, :], int64[:], int64[:], int64[:]))
    def cu_cell_list(x, box, ibox, cell_list, cell_counts, cells, cell_max):
        pi = cuda.grid(1)
        if pi >= x.shape[0]:
            return
        # xi = cuda.local.array(ndim, dtype=float64)
        # for k in range(ndim):
        # xi[k] = x[pi, k]
        xi = x[pi]
        ic = cu_cell_index(xi, box, ibox)
        cells[pi] = ic
        index = cuda.atomic.add(cell_counts, ic, 1)
        if index < cell_list.shape[0]:
            cell_list[ic, index] = pi
        else:
            cuda.atomic.max(cell_max, 0, index + 1)

    return cu_cell_map, cu_cell_list

class clist:
    def __init__(self, x, box, r_cut, gpu=0, cell_guess=50):
        self.x = x
        self.n = x.shape[0]
        self.n_dim = x.shape[1]
        self.box = box
        self.rc = r_cut
        self.gpu = gpu
        self.ibox = np.asarray(np.floor(box / r_cut), dtype=np.int64)
        self.n_cell = int(np.multiply.reduce(self.ibox))
        self.cell_adj = np.ones(self.n_dim, dtype=np.int64) * 3
        self.tpb = 64
        self.bpg = int(self.n // self.tpb + 1)
        self.bpg_cell = int(self.n_cell // self.tpb + 1)
        self.cell_guess = cell_guess
        # self.situ_zero = np.zeros(1, dtype=np.int64)
        self.cu_cell_map, self.cu_cell_list = _gen_func(x.dtype, self.n_dim)
        self.p_cell_max = cuda.pinned_array((1,), dtype=np.int64)
        with cuda.gpus[self.gpu]:
            self.d_x = cuda.to_device(self.x)
            self.d_box = cuda.to_device(self.box)
            self.d_cells = cuda.device_array(self.n, dtype=np.int64)
            self.d_cell_map = cuda.device_array((self.n_cell, 3 ** self.n_dim), dtype=np.int64)
            self.d_ibox = cuda.to_device(self.ibox)
            self.d_cell_adj = cuda.to_device(self.cell_adj)
            self.cu_cell_map[self.bpg_cell, self.tpb](self.d_ibox, self.d_cell_adj, self.d_cell_map)
            self.d_cell_list = cuda.device_array((self.n_cell, self.cell_guess),
                                                 dtype=np.int64)
            self.d_cell_counts = cuda.device_array(self.n_cell, dtype=np.int64)
            self.d_cell_max = cuda.device_array(1, dtype=np.int64)
        self.update()

    def update(self):
        with cuda.gpus[self.gpu]:
            while True:
                cu_set_to_int[self.bpg_cell, self.tpb](self.d_cell_counts, 0)
                self.cu_cell_list[self.bpg, self.tpb](self.d_x,
                                                      self.d_box,
                                                      self.d_ibox,
                                                      self.d_cell_list,
                                                      self.d_cell_counts,
                                                      self.d_cells,
                                                      self.d_cell_max)
                self.d_cell_max.copy_to_host(self.p_cell_max)
                cuda.synchronize()
                if self.p_cell_max[0] > self.cell_guess:
                    self.cell_guess = self.p_cell_max[0]
                    self.cell_guess = self.cell_guess + 8 - (self.cell_guess & 7)
                    self.d_cell_list = cuda.device_array((self.n_cell, self.cell_guess),
                                                         dtype=np.int64)
                else:
                    break