import numpy as np
from ._msd import msd_square
from simpletraj.dcd.dcd import DCDReader  # or self-made dcd reader
from AutoCorrelation import vec_ac
from ._msd import msd

_BUFFER_SIZE = 200000000  # 2e8 coordinates for n_frames * n_particles


def msd_dcd(dcd_file):
    r"""
    :param dcd_file: str, dcd_file
    :return: np.ndarray, msd
    """
    dcd = DCDReader(dcd_file)
    if dcd.periodic:
        raise ValueError("Error, periodic data found!")
    n_samples = dcd.numatoms
    n_frames = dcd.numatoms
    _BUFFER = _BUFFER_SIZE // n_frames
    n_buffer = n_samples // _BUFFER
    ret = np.zeros((n_frames,))
    counter = 0
    for i in range(n_buffer):
        x = np.asarray([np.copy(_[counter:counter+_BUFFER]) for _ in dcd])
        counter += _BUFFER
        ret += (msd_square(x) - 2 * vec_ac(x, cum=True))
    x = np.asarray([np.copy(_[counter:]) for _ in dcd])
    ret += (msd_square(x) - 2 * vec_ac(x, cum=True))
    return ret / n_samples
