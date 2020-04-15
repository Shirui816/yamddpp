from math import ceil

import numpy as np
from numba import cuda

from utils import cu_cell_count, cu_cell_ind
from utils import cu_max_int, cu_set_to_int
from utils import cu_nl_strain


class MDSystem(object):

    def __init__(self, x, box, ts, rc, strain=None, gpu=0, list_p=True):
        self.n = x.shape[0]
        self.pos = np.asarray(x, dtype=np.float64)
        self.box = np.asarray(box, dtype=np.float64)
        self.time_step = int(ts)
        self.n_dim = x.shape[1]
        self.gpu = gpu
        self.strain = np.asarray(strain, dtype=np.float64)
        self._nc_p = 100
        self.rc = rc
        if strain is None:
            self.strain = np.eye(self.n_dim, dtype=np.float64)
        self.pos_ortho = self.pos.dot(np.linalg.inv(strain).T)  # cell_list does not change
        self.ibox = np.asarray(box / rc, dtype=np.int64)
        self.n_cell = np.multiply.reduce(self.ibox)
        self.cell_id = None
        self.cell_list = None
        self.cell_count = None
        self.d_cell_id = None
        self.d_cell_list = None
        self.d_cell_count = None
        self.nl = None
        self.nc = None
        self.d_nl = None
        self.d_nc = None
        self.gpu = gpu
        with cuda.gpus[gpu]:
            self.d_pos = cuda.to_device(x)
            self.d_pos_ortho = cuda.to_device(self.pos_ortho)
            self.d_box = cuda.to_device(box)
            self.d_strain = cuda.to_device(self.strain)
            self.d_ibox = cuda.to_device(self.ibox)
            self.d_cell_id = cuda.device_array((self.n,), dtype=np.int64)
            # self.d_cell_id = cupy.asarray(self.d_cell_id)
            # self.d_cell_id = cupy.zeros((self.n,), dtype=np.int64)
            self._device = cuda.get_current_device()
            self.tpb = self._device.WARP_SIZE
            self.bpg = ceil(self.n / self.tpb)
            if list_p:
                self.cu_cell_list()
                self.cu_neighbour_list()

    def cu_cell_list(self):
        cu_cell_ind[self.bpg, self.tpb](self.pos_ortho, self.d_box, self.d_ibox, self.d_cell_id)
        # self.d_cell_list = cupy.argsort(self.d_cell_id)  # could be used by cuda.jit
        self.cell_id = self.d_cell_id.copy_to_host()
        cuda.synchronize()
        self.cell_list = np.argsort(self.cell_id)
        self.cell_id = self.cell_id[self.cell_list]  # need to use the cpu to make the RadixSort
        self.d_cell_id = cuda.to_device(self.cell_id)
        self.d_cell_list = cuda.to_device(self.cell_list)
        self.d_cell_count = cuda.device_array((self.n_cell,), dtype=np.int64)
        bpg = ceil(self.n_cell / self.tpb)
        cu_cell_count[bpg, self.tpb](self.d_cell_id, self.d_cell_count)
        self.cell_list = self.d_cell_list.copy_to_host()
        self.cell_count = self.d_cell_count.copy_to_host()
        cuda.synchronize()

    def cu_neighbour_list(self):
        _nl = cu_nl_strain(self.n_dim)
        dim = np.ones(self.n_dim, dtype=np.int64)
        d_nc = cuda.device_array((self.n,), dtype=np.int64)
        d_nl = cuda.device_array((self.n, self._nc_p), dtype=np.int64)
        d_nc_max = cuda.device_array((1,), dtype=np.int64)
        while True:
            cu_set_to_int[self.bpg, self.tpb](d_nc, 0)
            _nl[self.bpg, self.tpb](
                self.d_pos_ortho, self.d_box, self.d_ibox, self.d_strain,
                self.rc, self.d_cell_list, self.d_cell_count, d_nl, d_nc, dim
            )
            cu_max_int[self.bpg, self.tpb](d_nc, d_nc_max)
            nc_max = d_nc_max.copy_to_host()
            cuda.synchronize()
            if nc_max[0] > self._nc_p:
                self._nc_p = nc_max[0]
                d_nl = cuda.device_array((self.n, self._nc_p), dtype=np.int64)
            else:
                break
        self.d_nc = d_nc
        self.d_nl = d_nl
        self.nl = self.d_nl.copy_to_host()
        self.nc = self.d_nc.copy_to_host()
        cuda.synchronize()
