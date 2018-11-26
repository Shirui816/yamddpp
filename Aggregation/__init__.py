from ._com import com
from ._cluster_dbscan import coarse_grained_cluster
import numpy as np


def pbc(p, d):
    return p - d * np.round(p / d)


def handle_clusters(clusters, pos, types, box):
    meta = open('cluster_meta.txt', 'w')
    fmt = '%04d %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n'
    for i, cluster in enumerate(clusters):
        p_cluster = pos[cluster]
        p_types = types[cluster]
        midpoint = com(p_cluster, box / 2, -box / 2, midpoint=True)
        p_cluster = pbc(p_cluster - midpoint, box)  # always in (midpoint-box/2, midpoint+box/2)
        p_cluster -= p_cluster.mean(axis=0)  # make com to be 0
        rg_tensor = p_cluster.T.dot(p_cluster) / p_cluster.shape[0]
        e, v = np.linalg.eig(rg_tensor)
        o.write(fmt % (i, p_cluster.shape[0], e[0], e[1], e[2],
                       v.T[0, 0], v.T[0, 1], v.T[0, 2],
                       v.T[1, 0], v.T[1, 1], v.T[1, 2],
                       v.T[2, 0], v.T[2, 1], v.T[2, 2]))
        xyz = open('%04d.xyz' % i, 'w')
        xyz.write('%d\nmeta\n' % (p_cluster.shape[0]))
        for __, _ in zip(p_types, p_cluster):
            xyz.write('%s %.4f %.4f %.4f\n' % (__, _[0], _[1], _[2]))
        xyz.close()
    meta.close()
