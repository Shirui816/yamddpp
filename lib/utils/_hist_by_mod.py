import numpy as np
from numba import jit


def hist_vec_by_r(x, dr, r_bin, r_max, middle=None):
    # middle is the index of array x where the corresponding
    # position is zero vector.
    r_max2 = r_max ** 2
    ret = np.zeros(int(r_max / r_bin) + 1, dtype=x.dtype)
    cter = np.zeros(ret.shape, dtype=np.float)
    if middle is None:
        middle = np.zeros(x.ndim, dtype=np.float)

    @jit(nopython=True)
    def _func(x, dr, r_bin, r_max2, ret, cter, middle):
        # wired bug: if x.shape = (6,6,6), dr=1/6, middle=(3,3,3), r_cut=0.5
        # then index = (5,5,4) gives modulus r be exactly 0.5; if we let
        # dr = 0.166666666666, the modulus would be 0.4999...
        for idx in np.ndindex(x.shape):
            rr = 0
            for jdx, m in zip(idx, middle):
                rr += ((jdx - m) * dr) ** 2
            if rr < r_max2:
                kdx = int(rr ** 0.5 / r_bin)
                ret[kdx] += x[idx]
                cter[kdx] += 1

    _func(x, dr, r_bin, r_max2, ret, cter, middle)
    cter[cter == 0] = 1
    return ret / cter

# for datas of all dimensions. x.shape == (500, 500, 500), r.shape == (3, 500). ~ 2.05s
# np.ndindex is in C order: ret
# p,_ = np.histogram(np.linalg.norm(np.asarray(list(np.ndindex(a.shape))), axis=-1),bins=50,range=(0,5), weights=x.ravel('C'))
# cter:
# p,_ = np.histogram(np.linalg.norm(np.asarray(list(np.ndindex(a.shape))), axis=-1),bins=50,range=(0,5))
