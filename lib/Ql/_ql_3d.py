from math import floor, atan2, sqrt, ceil, pi

import numba as nb
import numpy as np
from numba import cuda

from ..utils import cu_nl
from ._spherical_harmonics import sphHar


def ql(x, box, rc, ls=np.array([4, 6]), gpu=0):
    _d = (ls.shape[0], int(2 * ls.max() + 1))
    _dd = _d[0]
    ret = np.zeros((x.shape[0], ls.shape[0]), dtype=np.float64)

    @cuda.jit("void(float64[:,:], float64[:], float64, int64[:,:], int64[:], int64[:], float64[:,:])")
    def _ql(_x, _box, _rc, _nl, _nc, _ls, _ret):
        i = cuda.grid(1)
        if i >= _x.shape[0]:
            return
        Qveci = cuda.local.array(_d, nb.complex128)
        resi = cuda.local.array(_dd, nb.float64)
        for _ in range(_d[0]):
            resi[_] = 0
            for __ in range(_d[1]):
                Qveci[_, __] = 0 + 0j
        nn = 0.0
        for j in range(_nc[i] - 1):
            pj = _nl[i, j]
            for k in range(j + 1, _nc[i]):
                pk = _nl[i, k]
                dx = _x[pk, 0] - _x[pj, 0]
                dy = _x[pk, 1] - _x[pj, 1]
                dz = _x[pk, 2] - _x[pj, 2]
                dx = dx - _box[0] * floor(dx / _box[0] + 0.5)
                dy = dy - _box[1] * floor(dy / _box[1] + 0.5)
                dz = dz - _box[2] * floor(dz / _box[2] + 0.5)
                dr = sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                nn += 1.0
                phi = atan2(dy, dx)
                if phi < 0:
                    phi = phi + 2 * pi
                cosTheta = dz / dr
                for _l in range(_ls.shape[0]):
                    l = _ls[_l]
                    for m in range(-l, l + 1):
                        Qveci[_l, m + l] += sphHar(l, m, cosTheta, phi)
        if nn < 1.0:
            nn = 1.0
        for _ in range(_d[0]):
            for __ in range(_d[1]):
                resi[_] += abs(Qveci[_, __] / nn) ** 2
        for _ in range(_d[0]):
            _ret[i, _] = sqrt(resi[_] * 4 * pi / (2 * _ls[_] + 1))

    nl, nc, nc_p = cu_nl(x, box, rc, gpu=gpu)
    with cuda.gpus[gpu]:
        device = cuda.get_current_device()
        tpb = device.WARP_SIZE
        bpg = ceil(x.shape[0] / tpb)
        _ql[bpg, tpb](
            x, box, rc, nl, nc, ls, ret
        )
    return ret
