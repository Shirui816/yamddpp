from numba import jit
import numpy as np


@jit
def norm_vec_to_r(x, r, r_bin, r_max):
    r_max2 = r_max ** 2
    ret = np.zeros(int(r_max / r_bin) + 1, dtype=np.float)
    cter = np.zeros(ret.shape, dtype=np.float)
    for idx in np.ndindex(x.shape):
        rr = 0
        for j, jdx in enumerate(idx):
            rr += r[j, jdx] ** 2
        if rr < r_max2:
            kdx = int(rr ** 0.5 / r_bin)
            ret[kdx] += x[idx]
            cter[kdx] += 1
    cter[cter == 0] = 1
    return ret / cter

# for datas of all dimensions. x.shape == (500, 500, 500), r.shape == (3, 500). ~ 2.05s
