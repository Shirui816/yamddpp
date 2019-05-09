import numpy as np
from . import batch_dot

def pbc(r, d):
    return r - d * np.round(r/d)

def batchRgTensor(samples, boxes):
    r"""Vectorized universal function.
    :param samples: np.ndarray, (...,n_chains, n_monomers,n_dimensions)
    :param boxes: np.ndarray, (...,n_dimensions)
    :return: np.ndarray ret.
    """
    if samples.ndim <= 3: # (n_chains, n_monomers, n_dimensions), samples.ndim>3 means at least there was frame info
        raise ValueError("NO~~~Are you using multiple box values for 1 frame data?")
    else:
        boxes = np.expand_dims(np.expand_dims(boxes, -2), -3)
	    # boxes' dimension is always lower than samples by 2 (n_chains, n_monomers)
	    # e.g., (10, 3) for 10 frames (10, n_chains, n_monomers, 3) for sample
    chain_length = samples.shape[-2]
    samples = pbc(np.diff(samples, axis=-2), boxes).cumsum(axis=-2) # samples -> (..., n_chains, n-1 monomers, n_dim)
    com = np.expand_dims(samples.sum(axis=-2)/chain_length, -2) # com -> (..., n_chains, 1, n_dim)
    samples = np.append(-com, samples-com, axis=-2) # samples - com of rest n-1 monomers and -com for the 1st monomer
    rgTensors = batch_dot(np.swapaxes(samples, -2, -1), samples) / chain_length
    # batch_dot == np.einsum('...mp,...pn->...mn', np.swapaxes(samples, -2, -1), samples) -> (..., n_chains, n_dim, n_dim)
    # batch_dot == np.einsum('...mp,...pn->...mn', (..., n_chains, n_dim, n_monomers),  (..., n_chains, n_monomers, n_dim))
    # batch_dot is way more faster than np.einsum 
    return np.linalg.eigh(rgTensors) # work on last (..., M, M) matrices
