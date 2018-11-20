from numba import jit
import numpy as np
from numpy.core.multiarray import ndarray


@jit
def cell_id(x: np.ndarray, box: np.ndarray, ibox: np.ndarray) -> int:
    ind: ndarray = np.asarray(((x / box + 0.5) * ibox), dtype=np.int64)
    return ind[0] + ind[1] * ibox[0] + ind[2] * ibox[1] * ibox[0]


@jit
def i_cell(cid: np.ndarray, ibox: np.ndarray) -> int:  # ravel in Fortran way.
    ind: ndarray = np.asarray((cid + ibox) % ibox, dtype=np.int64)
    return ind[0] + ind[1] * ibox[0] + ind[2] * ibox[1] * ibox[0]


# TODO: all-dimension support for above 2 funcs.


@jit
def cell_neighbours(ic: np.ndarray, ibox: np.ndarray) -> np.ndarray:
    ret: ndarray = np.zeros(3 ** ibox.shape[0], dtype=np.int64)
    for i, ind in enumerate(np.ndindex((3,) * ibox.shape[0])):
        # iterator of unraveled n-dimensional indices.
        ind = np.asarray(ind) - 1
        ret[i] = i_cell(ind + ic, ibox)
    return ret


@jit
def box_map(box: np.ndarray, r_cut: "np.ndarray or float") -> tuple:
    ibox: ndarray = np.asarray(box / r_cut, dtype=np.int64)
    ret: ndarray = np.zeros((np.multiply.reduce(ibox), 3 ** box.shape[0]))
    for ind in np.ndindex(ibox):
        ind = np.asarray(ind)
        ic = i_cell(ind, ibox)
        ret[ic] = cell_neighbours(ind, ibox)
    return ret, ibox


@jit
def linked_cl(pos: np.ndarray, box: np.ndarray, ibox: np.ndarray) -> tuple:
    head: ndarray = np.zeros(np.multiply.reduce(ibox), dtype=np.int64) - 1
    body: ndarray = np.zeros(pos.shape[0], dtype=np.int64)
    for i in range(pos.shape[0]):
        ic = cell_id(pos[i], box, ibox)
        body[i] = head[ic]
        head[ic] = i
    return head, body
