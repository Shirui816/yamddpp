#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: shirui <shirui816@gmail.com>

import numpy as np
from .cython_ import to_sq


def scatter(x, x_range, bins, q_bin, q_max, zero_padding=1):
    if x.shape[1] != 3:
        raise ValueError("3D coordinates are currently supported only!")
    assert x_range.shape[0] == x.shape[1]
    rho, edge = np.histogramdd(x, bins=bins, range=x_range)
    box = np.array(np.array([_[1] - _[0] for _ in x_range]))
    z_bins = (bins * zero_padding).astype(np.int64)
    _rft_sq = np.fft.rfftn(rho, s=z_bins)
    _sq = np.concatenate([_rft_sq, _rft_sq.conj()[:, :, ::-1][[0] +
                          list(range(z_bins[0] - 1, 0, -1)), :,
                          (z_bins[-1] + 1) % 2:-1][:, [0] +
                          list(range(z_bins[1] - 1, 0, -1)), :]],
                         axis=-1)
    _sq = np.abs(_sq) ** 2
    _d = box / bins
    q = np.vstack([np.fft.fftfreq(_sq.shape[_], _d[_]) for _ in range(3)])
    q = q * 2 * np.pi
    return to_sq(_sq, q, q_max, q_bin)
