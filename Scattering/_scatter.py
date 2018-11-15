#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: shirui <shirui816@gmail.com>

import numpy as np
from utils import norm_to_vec_cu
from utils import norm_vec_to_r


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
    # expand density with periodic data, enlarge sample periods.
    _rft_sq_y = _rft_sq_x
    if mode == 'ab':
        rho_y, edge = np.histogramdd(y, bins=bins, range=x_range)
        rho_y = np.pad(rho_y, [(0, _ * __) for _, __ in zip(rho_y.shape, expand)], 'wrap')
        _rft_sq_y = np.fft.rfftn(rho_y, s=z_bins)
    _rft_sq_xy = _rft_sq_x.conj() * _rft_sq_y
    _sq_xy = np.concatenate([_rft_sq_xy,
                             _rft_sq_xy.conj()
                             [:, :, ::-1]
                             [-np.arange(z_bins[0]), :, (z_bins[-1] + 1) % 2:-1]
                             [:, -np.arange(z_bins[1]), :]],
                            axis=-1)
    q = np.vstack([np.fft.fftfreq(_sq_xy.shape[_], _d[_]) for _ in range(_d.shape[0])])
    q = q * 2 * np.pi
    if use_gpu is False:
        return norm_vec_to_r(_sq_xy, q, q_bin, q_max)
    return norm_to_vec_cu(_sq_xy, q, q_bin, q_max, gpu=use_gpu)
