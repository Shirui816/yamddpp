#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: shirui <shirui816@gmail.com>

import numpy as np
from utils import norm_to_vec_cu
from utils import norm_vec_to_r


def rdf_xy(x, y, x_range, bins, r_bin=0.2, use_gpu=False):
    if not (isinstance(use_gpu, bool) or isinstance(use_gpu, int)):
        raise ValueError("`use_gpu' should be bool or int!")
    mode = 'ab' if x is not y else 'aa'
    box = np.array(np.array([_[1] - _[0] for _ in x_range]))
    px, ex = np.histogramdd(x, bins=bins, range=x_range)
    _ft_px = np.fft.rfftn(px)
    _ft_py = _ft_px
    if mode == 'ab':
        py, ex = np.histogramdd(y, bins=bins, range=x_range)
        _ft_py = np.fft.rfftn(py)
    _ft_px_py = _ft_px * _ft_py.conj()
    _rdf_xyz = np.fft.irfftn(_ft_px_py, bins)
    # _ft_py_px[t] == _ft_px_py[-t]
    _rdf_xyz[0, 0, 0] -= 0 if mode == 'ab' else x.shape[0]
    _rdf_xyz = np.fft.fftshift(_rdf_xyz)  # for x, y are in (-box/2, box/2)
    _r = np.vstack([_[:-1] + 0.5 * (_[-1] - _[-2]) for _ in ex])
    if use_gpu is False:
        _rdf = norm_vec_to_r(_rdf_xyz, _r, r_bin, box.min() / 2)
    else:
        _rdf = norm_to_vec_cu(_rdf_xyz, _r, r_bin, box.min() / 2,
                              gpu=use_gpu)
    _rdf /= x.shape[0] * y.shape[0]
    _rdf *= np.multiply.reduce(bins)
    return (np.arange(_rdf.shape[0]) + 0.5) * r_bin, _rdf
