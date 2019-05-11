from . import bondVecs
from . import batch_inner_prod
import numpy as np

def bondAngles(samples: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    r"""Batch calculation of Rg Tensors.
    :param samples: np.ndarray, (...,n_chains, n_monomers,n_dim)
    e.g. (n_batch, n_frames, n_chains, n..., n_dim)
    :param boxes: np.ndarray, (...,n_dimensions),
    e.g. (n_batch, n_frames, n_dim)
    :return: np.ndarray ret.
    """
    bond_vecs = bondVecs(samples, boxes)
    prod = np.einsum(
        '...ij,...ij->...i', bond_vecs[..., :-1, :], bond_vecs[..., 1:, :]
    )
    norm = np.linalg.norm(bond_vecs, axis=-1)
    return np.arccos(np.clip(prod / norm[..., :-1] / norm[..., 1:], -1, 1))


def bondAngles_guv(samples: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    r"""Batch calculation of Rg Tensors.
    :param samples: np.ndarray, (...,n_chains, n_monomers,n_dim)
    e.g. (n_batch, n_frames, n_chains, n..., n_dim)
    :param boxes: np.ndarray, (...,n_dimensions),
    e.g. (n_batch, n_frames, n_dim)
    :return: np.ndarray ret.
    This version should be faster. ^_^
    """
    bond_vecs = bondVecs(samples, boxes)
    prod = batch_inner_prod(bond_vecs[..., :-1, :], bond_vecs[..., 1:, :])
    return np.arccos(np.clip(prod, -1, 1))
