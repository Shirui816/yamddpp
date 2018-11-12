#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: shirui <shirui816@gmail.com>

from numba import cuda
from math import floor
from math import sqrt
import numpy as np


@cuda.jit("void(float32[:,:,:], float32[:, :], float64, float64, float32[:], float32[:], float32[:])")
def cu_hist_xyz_to_r_real(m_xyz, r, r_max, r_bin, m_r, rs, ct):
    i, j, k = cuda.grid(3)
    if i >= m_xyz.shape[0]:
        return
    if j >= m_xyz.shape[1]:
        return
    if k >= m_xyz.shape[2]:
        return
    x = r[0, i]
    y = r[1, j]
    z = r[2, k]
    r_ = sqrt(x * x + y * y + z * z)
    if r_ < r_max:
        idx = int(floor(r_ / r_bin))
        cuda.atomic.add(m_r, idx, m_xyz[i, j, k])
        cuda.atomic.add(rs, idx, r_)
        cuda.atomic.add(ct, idx, 1.0)


@cuda.jit("void(float32[:,:,:], float32[:,:,:], float32[:, :], float64, float64, float32[:],"
          "float32[:], float32[:], float32[:])")
def cu_hist_xyz_to_r_comp(m_xyz_r, m_xyz_i, r, r_max, r_bin, m_r_r, m_r_i, rs, ct):
    i, j, k = cuda.grid(3)
    if i >= m_xyz_r.shape[0]:
        return
    if j >= m_xyz_r.shape[1]:
        return
    if k >= m_xyz_r.shape[2]:
        return
    x = r[0, i]
    y = r[1, j]
    z = r[2, k]
    r_ = sqrt(x * x + y * y + z * z)
    if r_ < r_max:
        idx = int(floor(r_ / r_bin))
        cuda.atomic.add(m_r_r, idx, m_xyz_r[i, j, k])
        cuda.atomic.add(m_r_i, idx, m_xyz_i[i, j, k])
        cuda.atomic.add(rs, idx, r_)
        cuda.atomic.add(ct, idx, 1.0)


def hist_xyz_to_r(m_xyz, r, r_max, r_bin, gpu=0):
    with cuda.gpus[gpu]:
        device = cuda.get_current_device()
        tpb = (device.WARP_SIZE,) * 3
        bpg = tuple((int(np.ceil(float(_) / __))
                     for _, __ in zip(m_xyz.shape, tpb)))
        n = int(floor(r_max / r_bin))
        m_r = np.zeros((n,), dtype=np.float32)
        rs = np.zeros((n,), dtype=np.float32)
        ct = np.zeros((n,), dtype=np.float32)
        if m_xyz.dtype == np.float32:
            cu_hist_xyz_to_r_real[bpg, tpb](m_xyz.astype(np.float32), r.astype(np.float32),
                                            r_max, r_bin, m_r, rs, ct)
        else:
            m_r_i = np.zeros((n,), dtype=np.float32)
            cu_hist_xyz_to_r_comp[bpg, tpb](m_xyz.real.astype(np.float32),
                                            m_xyz.imag.astype(np.float32),
                                            r.astype(np.float32), r_max,
                                            r_bin, m_r, m_r_i, rs, ct)
            m_r = m_r + 1j * m_r_i
    ct[ct == 0] = 1
    return rs / ct, m_r / ct
