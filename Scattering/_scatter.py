#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: shirui <shirui816@gmail.com>

import numpy as np
from .c_libs import hist_xyz_to_r
from .numba_cuda import hist_xyz_to_r as cu_hist_xyz_to_r


def scatter_xy(x, y, x_range, bins, q_bin, q_max, zero_padding=1, expand=0, use_gpu=False):
    mode = 'ab' if x is not y else 'aa'
    bins = np.asarray(bins)
    x_range = np.asarray(x_range)
    expand = np.asarray(expand)
    if x.shape[1] != 3:
        raise ValueError("3D coordinates are currently supported only!")
    if bins.ndim < 1:
        bins = np.asarray([bins] * 3)
    if not (isinstance(use_gpu, bool) or isinstance(use_gpu, int)):
        raise ValueError("`use_gpu' should be bool or int!")
    rho_x, edge = np.histogramdd(x, bins=bins, range=x_range)
    if expand.ndim < 1:
        expand = np.asarray([expand] * rho_x.ndim)
    box = np.array(np.array([_[1] - _[0] for _ in x_range]))
    _d = box / bins
    z_bins = (np.asarray(rho_x.shape) * zero_padding).astype(np.int64)
    rho_x = np.pad(rho_x, [(0, _ * __) for _, __ in zip(rho_x.shape, expand)], 'wrap')
    _rft_sq_x = np.fft.rfftn(rho_x, s=z_bins)
    _sq_x = np.concatenate([_rft_sq_x, _rft_sq_x.conj()[:, :, ::-1][-np.arange(z_bins[0]), :,
                                       (z_bins[-1] + 1) % 2:-1][:, -np.arange(z_bins[1]), :]],
                           axis=-1)
    # expand density with periodic data, enlarge sample periods.
    _sq_y = _sq_x
    if mode == 'ab':
        rho_y, edge = np.histogramdd(y, bins=bins, range=x_range)
        rho_y = np.pad(rho_y, [(0, _ * __) for _, __ in zip(rho_y.shape, expand)], 'wrap')
        _rft_sq_y = np.fft.rfftn(rho_y, s=z_bins) if mode == 'ab' else _rft_sq_x
        _sq_y = np.concatenate([_rft_sq_y, _rft_sq_y.conj()[:, :, ::-1][-np.arange(z_bins[0]), :,
                                           (z_bins[-1] + 1) % 2:-1][:, -np.arange(z_bins[1]), :]],
                               axis=-1)
    _sq_xy = _sq_x.conj() * _sq_y
    q = np.vstack([np.fft.fftfreq(_sq_xy.shape[_], _d[_]) for _ in range(_d.shape[0])])
    q = q * 2 * np.pi
    if use_gpu is False:
        return hist_xyz_to_r(_sq_xy, q, q_max, q_bin)
    return cu_hist_xyz_to_r(_sq_xy, q, q_max, q_bin, gpu=use_gpu)
