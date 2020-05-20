import numpy as np

from . import batch_dot
from . import bond_vecs


def batch_rg_tensor(samples: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    r"""Batch calculation of Rg Tensors.
    :param samples: np.ndarray, (...,n_chains, n_monomers,n_dim)
    e.g. (n_batch, n_frames, n_chains, n..., n_dim)
    :param boxes: np.ndarray, (...,n_dimensions),
    e.g. (n_batch, n_frames, n_dim)
    :return: np.ndarray ret.
    """
    samples = bond_vecs(samples, boxes).cumsum(axis=-2)
    samples -= np.expand_dims(samples.mean(axis=-2), -2)
    rg_tensors = batch_dot(np.swapaxes(samples, -2, -1), samples)
    return rg_tensors / samples.shape[-2]
