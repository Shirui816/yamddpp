from math import floor, atan2, sqrt, pi

import numba as nb
import numpy as np
from numba import cuda

from ._spherical_harmonics import sphHar
from ..plists import nlist


class ql:
    def __init__(self, frame, ls=np.asarray([4, 6])):
        self.frame = frame
        self.gpu = frame.gpu
        self.ls = ls
        self._qvi = (ls.shape[0], int(2 * ls.max() + 1))
        self._rei = ls.shape[0]
        self.cu_ql_local = self._ql_local_func()
        self.cu_ql_avg = self._ql_avg_func()
        self.nlist = nlist(self.frame, contain_self=1)

    def update(self, x=None, box=None, rc=None):
        if x is not None:
            self.frame.x = x
        if box is not None:
            self.frame.box = box
        if rc is not None:
            self.frame.rc = rc
        self.frame.update()
        self.nlist.update()
        self.calculate('all')

    def calculate(self, mode='all'):
        with cuda.gpus[self.gpu]:
            d_ls = cuda.to_device(self.ls)
            device = cuda.get_current_device()
            tpb = device.WARP_SIZE
            bpg = int(np.ceil(self.frame.x.shape[0] / tpb))
            if mode == 'all' or mode == 'local':
                self.ql_local = np.zeros((self.frame.x.shape[0], self.ls.shape[0]), dtype=np.float64)
                d_ql_local = cuda.to_device(self.ql_local)
                self.cu_ql_local[bpg, tpb](
                    self.frame.d_x,
                    self.frame.d_box,
                    self.frame.rc,
                    self.nlist.d_nl,
                    self.nlist.d_nc,
                    d_ls,
                    d_ql_local
                )
                d_ql_local.copy_to_host(self.ql_local)
                cuda.synchronize()
            if mode == 'all' or mode == 'avg':
                self.ql_avg = np.zeros(self.ls.shape[0])
                Qvec = np.zeros((self.x.shape[0], self.ls.shape[0], int(2 * self.ls.max() + 1)), dtype=np.complex128)
                d_Qvec = cuda.to_device(Qvec)
                n_bonds = np.zeros(1, dtype=np.int64)
                d_n_bonds = cuda.to_device(n_bonds)
                self.cu_ql_avg[bpg, tpb](
                    self.frame.d_x,
                    self.frame.d_box,
                    self.frame.rc,
                    self.nlist.d_nl,
                    self.nlist.d_nc,
                    self.ls,
                    d_Qvec,
                    d_n_bonds
                )
                d_n_bonds.copy_to_host(n_bonds)
                d_Qvec.copy_to_host(Qvec)
                cuda.synchronize()
                Qvec = np.sum(Qvec, axis=0)
                self.n_bonds = n_bonds[0]
                if self.n_bonds < 1.0:
                    self.n_bonds = 1.0
                for i in range(Qvec.shape[0]):
                    tmp = 0
                    for j in range(Qvec.shape[1]):
                        tmp += abs(Qvec[i, j] / self.n_bonds) ** 2
                    self.ql_avg[i] = sqrt(tmp * 4 * np.pi / (2 * self.ls[i] + 1))

    def _ql_local_func(self):
        _qvi = self._qvi
        _rei = self._rei

        @cuda.jit("void(float64[:,:], float64[:], float64, int64[:,:], int64[:], int64[:], float64[:,:])")
        def _ql_local(x, box, rc, nl, nc, ls, ret):
            i = cuda.grid(1)
            if i >= x.shape[0]:
                return
            Qveci = cuda.local.array(_qvi, nb.complex128)
            resi = cuda.local.array(_rei, nb.float64)
            for _ in range(Qveci.shape[0]):
                resi[_] = 0
                for __ in range(Qveci.shape[1]):
                    Qveci[_, __] = 0 + 0j
            nn = 0.0
            for j in range(nc[i] - 1):
                pj = nl[i, j]
                for k in range(j + 1, nc[i]):
                    pk = nl[i, k]
                    dx = x[pk, 0] - x[pj, 0]
                    dy = x[pk, 1] - x[pj, 1]
                    dz = x[pk, 2] - x[pj, 2]
                    dx = dx - box[0] * floor(dx / box[0] + 0.5)
                    dy = dy - box[1] * floor(dy / box[1] + 0.5)
                    dz = dz - box[2] * floor(dz / box[2] + 0.5)
                    dr = sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    if dr >= rc:
                        continue
                    nn += 1.0
                    phi = atan2(dy, dx)
                    if phi < 0:
                        phi = phi + 2 * pi
                    cosTheta = dz / dr
                    for _l in range(ls.shape[0]):
                        l = ls[_l]
                        for m in range(-l, l + 1):
                            Qveci[_l, m + l] += sphHar(l, m, cosTheta, phi)
            # print(i, nn)
            if nn < 1.0:
                nn = 1.0
            for _ in range(Qveci.shape[0]):
                for __ in range(Qveci.shape[1]):
                    resi[_] += abs(Qveci[_, __] / nn) ** 2
            for _ in range(Qveci.shape[0]):
                ret[i, _] = sqrt(resi[_] * 4 * pi / (2 * ls[_] + 1))

        return _ql_local

    def _ql_avg_func(self):
        @cuda.jit(
            "void(float64[:,:], float64[:], float64, int64[:,:], int64[:], int64[:], complex128[:,:,:], int64[:])")
        def _ql_avg(x, box, rc, nl, nc, ls, Qvec, n_bonds):
            i = cuda.grid(1)
            if i >= x.shape[0]:
                return
            nn = 0
            for j in range(nc[i]):
                pj = nl[i, j]
                if pj <= i:
                    continue
                dx = x[pj, 0] - x[i, 0]
                dy = x[pj, 1] - x[i, 1]
                dz = x[pj, 2] - x[i, 2]
                dx = dx - box[0] * floor(dx / box[0] + 0.5)
                dy = dy - box[1] * floor(dy / box[1] + 0.5)
                dz = dz - box[2] * floor(dz / box[2] + 0.5)
                dr = sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                if dr >= rc:
                    continue
                nn += 1
                phi = atan2(dy, dx)
                if phi < 0:
                    phi = phi + 2 * pi
                cosTheta = dz / dr
                for _l in range(ls.shape[0]):
                    l = ls[_l]
                    for m in range(-l, l + 1):
                        Qvec[i, _l, m + l] += sphHar(l, m, cosTheta, phi)  # thread-safe
            cuda.atomic.add(n_bonds, 0, nn)

        return _ql_avg
