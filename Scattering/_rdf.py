#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: shirui <shirui816@gmail.com>

import numpy as np
from .cython_ import xyz_to_r


def rdf(x, y, box, bins, r_bin=0.2):
    px, ex = np.histogramdd(x, bins=bins,
                            range=[(-_/2, _/2) for _ in box])
    py, ex = np.histogramdd(y, bins=bins,
                            range=[(-_/2, _/2) for _ in box])
    _ft_px = np.fft.rfftn(px)
    _ft_py = np.fft.rfftn(py)
    _ft_px_py = _ft_px * _ft_py.conj()
    _rdf_xyz = np.fft.irfftn(_ft_px_py)
    # _ft_py_px[t] == _ft_px_py[-t]
    _rdf_xyz[0, 0, 0] = 0
    _rdf_xyz = np.fft.fftshift(_rdf_xyz)  # for x, y are in (-box/2, box/2)
    _r = np.vstack([_[:-1] + 0.5 * (_[-1] - _[-2]) for _ in ex])
    _rdf = xyz_to_r(_rdf_xyz, _r, box.min()/2, r_bin)
    _rdf /= x.shape[0] * y.shape[0]
    _rdf *= np.multiply.reduce(bins)
    return (np.arange(_rdf.shape[0]) + 0.5) * r_bin, _rdf
