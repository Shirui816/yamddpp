from numba import jit
import numpy as np


@jit
def cell_id(x, box, ibox):
    ind = np.asarray(((x / box + 0.5) * ibox), dtype=np.int64)
    return ind[0] + ind[1] * ibox[0] + ind[2] * ibox[1] * ibox[0]


@jit
def i_cell(cid, ibox):
    ind = np.asarray((cid + ibox) % ibox, dtype=np.int64)
    return ind[0] + ind[1] * ibox[0] + ind[2] * ibox[1] * ibox[0]

# TODO: all-dimension support for above 2 funcs.

@jit
def cell_neighbours(ic, ibox):
    ret = np.zeros(3 ** ibox.shape[0], dtype=np.int64)
    ct = 0
    for ind in np.ndindex((3,) * ibox.shape[0]):
        ind = np.asarray(ind) - 1
        ret[ct] = i_cell(ind + ic, ibox)
        ct += 1
    return ret


@jit
def box_map(box, r_cut):
    ibox = np.asarray(box / r_cut, dtype=np.int64)
    ret = np.zeros((np.multiply.reduce(ibox), 3 ** box.shape[0]))
    for ind in np.ndindex(ibox):
        ind = np.asarray(ind)
        ic = i_cell(ind, ibox)
        ret[ic] = cell_neighbours(ind, ibox)
    return ret, ibox


@jit
def linked_cl(pos, box, ibox):
    head = np.zeros(np.multiply.reduce(ibox), dtype=np.int64) - 1
    body = np.zeros(pos.shape[0], dtype=np.int64)
    for i in range(pos.shape[0]):
        ic = cell_id(pos[i], box, ibox)
        body[i] = head[ic]
        head[ic] = i
    return head, body
