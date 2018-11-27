from ._com import com
from utils import linked_cl
from utils import get_from_cell
import numpy as np


def pbc(p, d):
    return p - d * np.round(p / d)


def handle_clusters(clusters, pos, types, box, bins=50):
    r"""Handle the results of clustering.
    :param clusters: list, list of clusters/
    :param pos: np.ndarray, positions
    :param types: np.ndarray, types
    :param box: np.ndarray, box
    :param bins: int, bins to check percolation.
    :return: None
    """
    meta = open('cluster_meta.txt', 'w')
    fmt = '%04d %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n'
    for i, cluster in enumerate(clusters):
        p_cluster = pos[cluster]
        p_types = types[cluster]
        percolate = np.asarray(
            [np.all(np.histogram(_, bins=bins, range=(-__ / 2, __ / 2))[0] > 0) for _, __ in zip(p_cluster.T, box)],
            dtype=np.bool)
        # center_of_mass = com(p_cluster, box / 2, -box / 2)
        # midpoint = com(p_cluster, box / 2, -box / 2, midpoint=True)
        # midpoint[percolate] = center_of_mass[percolate]  # using midpoint if not percolate, else center of mass
        percolate = np.logical_not(percolate)  # midpoint=True for not percolate
        midpoint = np.asarray([com(coors, box / 2, -box / 2, midpoint=percolate_p) for coors, percolate_p in
                               zip(p_cluster.T, percolate)])  # using midpoint=True if not percolate, else com
        # calculate in each dimension, faster.
        p_cluster = pbc(p_cluster - midpoint, box)  # always in (midpoint-box/2, midpoint+box/2)
        p_cluster -= p_cluster.mean(axis=0)  # make com to be 0
        rg_tensor = p_cluster.T.dot(p_cluster) / p_cluster.shape[0]
        e, v = np.linalg.eig(rg_tensor)
        meta.write(fmt % (i, p_cluster.shape[0], e[0], e[1], e[2],
                          v.T[0, 0], v.T[0, 1], v.T[0, 2],
                          v.T[1, 0], v.T[1, 1], v.T[1, 2],
                          v.T[2, 0], v.T[2, 1], v.T[2, 2]))
        xyz = open('%04d.xyz' % i, 'w')
        xyz.write('%d\nmeta\n' % (p_cluster.shape[0]))
        for __, _ in zip(p_types, p_cluster):
            xyz.write('%s %.4f %.4f %.4f\n' % (__, _[0], _[1], _[2]))
        xyz.close()
        meta.close()


def coarse_grained_cluster(pos, box, func, kwargs=None, r_cut=0):
    r"""Cluster particles directly or cluster the cells to reduce calculation.

    :param pos: np.ndarray, positions
    :param box: np.ndarray, box
    :param func: callable, clustering method
    :param kwargs: dictionary, args of func
    :param r_cut: float or np.ndarray, coarse-grain size, 0 for clustering directly.
    :return: list, clusters with particles ids.
    """
    if kwargs is None:
        kwargs = {}
    if not r_cut:
        return func(pos, box, **kwargs)
    bins = np.asarray(box / r_cut, dtype=np.int)
    # weights, _ = np.histogramdd(pos, bins=bins, range=[(-_ / 2, _ / 2) for _ in box])
    # weights = weights.ravel(order='F')  # ravel in Fortran
    # build the cell-list at the same time
    head, body, weights = linked_cl(pos, box, bins)  # weights was already raveled in Fortran way
    coordinates = np.vstack(np.unravel_index(np.arange(weights.shape[0]), bins, order='F')).T
    coordinates = coordinates * r_cut
    fitted = func(coordinates, box, **kwargs)
    clusters = [np.arange(head.shape[0])[fitted == _]
                for _ in list(set(fitted)) if _ != -1]  # cell-ids of cells in one cluster
    ret = []
    for cells in clusters:  # cluster consists of cells, for cluster in clusters
        tmp = []
        for cell in cells:  # for cell in a cluster of cells
            tmp.extend(get_from_cell(cell, head, body))  # find particles ids in the cell and add to tmp
        ret.append(tmp)
    return ret  # return the particle ids
