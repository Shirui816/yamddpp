#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: shirui <shirui816@gmail.com>

from .functions import real_input
from .functions import complex_input
import numpy as np

def hist_xyz_to_r(m_xyz, r, r_max, r_bin):
    if m_xyz.dtype == np.float64:
        return real_input(m_xyz, r, r_max, r_bin)
    return complex_input(m_xyz, r, r_max, r_bin)
