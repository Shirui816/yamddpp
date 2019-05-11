from numba import float64
from numba import guvectorize
from Aggregation import pbc
from ._rouse_modes import normal_modes
from ._rg_tensor import batchRgTensor


def bondVecs(samples: np.ndarray, boxes: np.ndarray) -> np.ndarray:
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
    bond_vecs = pbc(
        np.diff(samples, axis=-2, prepend=samples[..., :1, :]), boxes
    )
    return bond_vecs


@guvectorize([(float64[:, :], float64[:, :], float64[:, :])], '(n,p),(p,m)->(n,m)',
             target='parallel')  # target='cpu','gpu'
def batch_dot(a, b, ret):  # much more faster than np.tensordot or np.einsum
    r"""Vectorized universal function.
    :param a: np.ndarray, (...,N,P)
    :param b: np.ndarray, (...,P,M)
    axes will be assigned automatically to last 2 axes due to the signatures.
    this functions is actual np.einsum('...mp,....pn->...mn', a, b), or
    np.matmul(a, b). But this is much faster.
    :return: np.ndarray, results. (...,N,M)
    """
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            tmp = 0.
            for k in range(a.shape[1]):
                tmp += a[i, k] * b[k, j]
        ret[i, j] = tmp


@guvectorize([(float64[:], float64[:], float64[:])],'(n),(n)->()')
def batch_inner_prod(a, b, ret):
	tmp1 = tmp2 = tmp3 = 0
	for i in range(a.shape[0]):
		tmp1 += a[i] * b[i]
		tmp2 += a[i] * a[i]
		tmp3 += b[i] * b[i]
	ret[0] = tmp1 / (tmp2 * tmp3) ** 0.5

# TODO: CReTA
