#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: shirui <shirui816@gmail.com>

import numpy as np

# Add cython version


def to_sq_python(s_q, freqx, freqy, freqz, qmax, qbin, sq, qs, ct):
    for i in range(1, s_q.shape[0]):
        qx = freqx[i]
        for j in range(1, s_q.shape[1]):
            qy = freqy[j]
            for k in range(1, s_q.shape[2]):  # Zero-freq is meaningless
                qz = freqz[k]
                q = (qx * qx + qy * qy + qz * qz) ** 0.5
                if q < qmax:
                    idx = int(q / qbin)
                    sq[idx] += s_q[i, j, k]
                    qs[idx] += q
                    ct[idx] += 1
    ct[ct == 0] = 1
    return qs / ct, sq / ct


class Scattering(object):
    def __init__(self, r_matrix, r_range, bins_, q_bin, q_max,
                 debug=False, zero_padding=1):
        r"""
        :param r_matrix: np.ndarray for (n_particles, n_dimension)
        :param r_range: np.ndarray for boxsize
        :param bins_: integer or np.ndarray for bins of boxsize
        :param q_bin: double, binsize of q
        :param q_max: double, maximum of q
        :param debug: bool, debug info
        :param zero_padding: integer or np.ndarray, 0 padding number on all demension
        """
        fft = np.fft
        if bins_.shape[0] != r_matrix.shape[-1]:
            raise (ValueError("Bins must have same dimension with positions!"))
        self.rhoMatrix, edge = np.histogramdd(r_matrix, bins=bins_, range=r_range)
        box = np.array([_[1] - _[0] for _ in r_range])
        self.q_norm = 2 * np.pi / box / zero_padding
        self.d = box / bins_
        z_bins_ = (bins_ * zero_padding).astype(np.int64)
        r_sq = fft.rfftn(self.rhoMatrix,
                         s=z_bins_.astype(np.int64))
        # using 0-padding to enhance freq-domain sampling.
        self.SQ = np.concatenate([r_sq, r_sq.conj()[:, :, ::-1][[0] +
                                  list(range(z_bins_[0] - 1, 0, -1)), :,
                                  (z_bins_[-1] + 1) % 2:-1][:, [0] +
                                  list(range(z_bins_[1] - 1, 0, -1)), :]],
                                 axis=-1)
        # Don't ask why. Currently only 3-D is supported here. Unless this index
        # method could be generalized.
        if debug:
            print(np.isclose(self.SQ, fft.fftn(self.rhoMatrix, s=z_bins_)).all())
        self.qbin = q_bin
        self.qmax = q_max
        self.SQ = abs(self.SQ) ** 2

    def to_sq(self):
        freqx = np.fft.fftfreq(self.SQ.shape[0], self.d[0]) * 2 * np.pi
        freqy = np.fft.fftfreq(self.SQ.shape[1], self.d[1]) * 2 * np.pi
        freqz = np.fft.fftfreq(self.SQ.shape[2], self.d[2]) * 2 * np.pi
        sq = np.zeros((int(self.qmax / self.qbin),))
        qs = np.zeros((int(self.qmax / self.qbin),))
        ct = np.zeros((int(self.qmax / self.qbin),))
        return to_sq_python(self.SQ, freqx, freqy, freqz, self.qmax, self.qbin, sq, qs, ct)
