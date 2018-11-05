from numba import cuda
from numba import jit
import numpy as np
from math import floor
from math import ceil
from pyculib.sorting import RadixSort


@cuda.jit("int64(float64[:], float64[:], int64[:])", device=True)
def cu_cell_id(p, box, ibox):
    return floor((p[0] / box[0] + 0.5) * ibox[0]) + floor((p[1] / box[1] + 0.5) * ibox[1]) * ibox[0] + \
           floor((p[2] / box[2] + 0.5) * ibox[2]) * ibox[1] * ibox[0]


@cuda.jit("void(float64[:, :], float64[:], int64[:], int64[:]")
def cu_cell_ind(pos, box, ibox, ret):
    i = cuda.grid(1)
    if i < pos.shape[0]:
        pi = pos[i]
        ic = cu_cell_id(pi, box, ibox)
        ret[i] = ic


@jit
def count(cell_id):
    n_cell = cell_id.max()
    ret = np.zeros(n_cell + 1)
    for i in range(cell_id.shape[0]):
        ret[cell_id[i] + 1] += 1
    return np.cumsum(ret)


def cu_cell_list(pos, box, ibox, gpu=0):
    n = pos.shape[0]
    sorter = RadixSort(n, np.int64)
    cell_id = np.zeros(n).astype(np.int64)
    cell_list = np.arange(n).astype(np.int64)
    with cuda.gpus[gpu]:
        device = cuda.get_current_device()
        tpb = device.WARP_SIZE
        bpg = ceil(n / tpb)
        cu_cell_ind[bpg, tpb](pos, box, ibox, cell_id)
        sorter.sort(keys=cell_id, vals=cell_list)
    cell_count = count(cell_id)
    return cell_list, cell_count

# calling: for a particle in nth cell, cell_count[n] gives the start index of
# cell_list, and cell_count[n+1] gives the end index of cell_list.
# e.g. for 0th cell, cell_count[0] = 0 and cell_count[1] = number_of_particles_in_0th_cell
# for ith cell, cell_count[i] = number_of_particles_of_all_(0, i-1)th_cell and cell_count[i+1]
# = number_of_particles_of_all_(0, i)th_cell. For cell_list is sorted by cell_ids:
# cell_list = [pid_in_0th_cell, pid_in_0th_cell, ..., pid_in_nth_cell]
# cell_list[number_of_particles_of_all_(0, i-1)th_cell] = 1st_pid_in_ith_cell
