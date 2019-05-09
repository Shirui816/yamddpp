import numpy as np
from . import batch_dot


def pbc(r, d):
    return r - d * np.round(r/d)


def batchRgTensor(samples, boxes):
    r"""Batch calculation of Rg Tensors.
    :param samples: np.ndarray, (...,n_chains, n_monomers,n_dimensions)
    :param boxes: np.ndarray, (...,n_dimensions)
    :return: np.ndarray ret.
    """
    if samples.ndim <= 3:
        raise ValueError("NO~~~Are you using multiple box values for 1 frame data?")
    else:
        boxes = np.expand_dims(np.expand_dims(boxes, -2), -3)
    chain_length = samples.shape[-2]
    samples = pbc(np.diff(samples, axis=-2), boxes).cumsum(axis=-2)
    com = np.expand_dims(samples.sum(axis=-2)/chain_length, -2)
    samples = np.append(-com, samples-com, axis=-2)
    rgTensors = batch_dot(np.swapaxes(samples, -2, -1), samples) / chain_length
    return np.linalg.eigh(rgTensors)
