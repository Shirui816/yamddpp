import numpy as np
from ._msd import msd_square
from simpletraj.dcd.dcd import DCDReader  # or self-made dcd reader
from ._msd import msd
from ._mqd import mqd

_BUFFER_SIZE = 200000000  # 2e8 coordinates for n_frames * n_particles


def traj_dcd(dcd_file, func=msd, cum=True):
    r"""
    :param dcd_file: str, dcd_file
    :param func: callable, msd or mqd
    :param cum: bool
    :return: np.ndarray, msd
    """
    dcd = DCDReader(dcd_file)
    if dcd.periodic:
        raise ValueError("Error, periodic data found!")
    n_samples = dcd.numatoms
    n_frames = dcd.numatoms
    _BUFFER = _BUFFER_SIZE // n_frames
    n_buffer = n_samples // _BUFFER
    _shape = (n_frames,) if cum else (n_frames, n_samples)
    ret = np.zeros(_shape)
    counter = 0
    for i in range(n_buffer):
        x = np.asarray([np.copy(_[counter:counter+_BUFFER]) for _ in dcd])
        counter += _BUFFER
        ret += func(x, cum=cum)
    x = np.asarray([np.copy(_[counter:]) for _ in dcd])
    ret += func(x, cum=cum)
    return ret / n_samples
