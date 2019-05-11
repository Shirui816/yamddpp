import numpy as np
from . import batch_dot
from Aggregation import pbc


def batchRgTensor(samples: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    r"""Batch calculation of Rg Tensors.
    :param samples: np.ndarray, (...,n_chains, n_monomers,n_dim)
    e.g. (n_batch, n_frames, n_chains, n..., n_dim)
    :param boxes: np.ndarray, (...,n_dimensions),
    e.g. (n_batch, n_frames, n_dim)
    :return: np.ndarray ret.
    """
    if samples.ndim == 2:
        samples = np.expand_dims(samples, 0)
    if samples.ndim < boxes.ndim + 2:
        raise ValueError(
            "Are you using multiple box values for an 1-frame sample?"
        )
    boxes = np.expand_dims(np.expand_dims(boxes, -2), -3)
    chain_length = samples.shape[-2]
    samples = pbc(
        np.diff(samples, axis=-2, prepend=samples[..., :1, :]), boxes
    ).cumsum(axis=-2)
    samples -= np.expand_dims(samples.mean(axis=-2), -2)
    rgTensors = batch_dot(np.swapaxes(samples, -2, -1), samples) / chain_length
    return rgTensors