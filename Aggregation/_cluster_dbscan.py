import numpy as np
from numba import cuda
from math import floor, sqrt, ceil
from sklearn.cluster import DBSCAN


# For usage of cuda.jit, run `conda install cudatoolkit=9.0` after the
# installation of Anaconda env. numba.cuda cannot compile under
# latest `cudatoolkit=9.2' currently.


@cuda.jit('float64(float64[:], float64[:], float64[:])', device=True)
def pbc_dist_cu(a, b, box):
    tmp = 0
    for i in range(a.shape[0]):
        d = b[i] - a[i]
        d = d - floor(d / box[i] + 0.5) * box[i]
        tmp += d * d
    return sqrt(tmp)


@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64[:,:])')
def pbc_pairwise_distance(x, y, box, ret):
    i, j = cuda.grid(2)
    if i >= x.shape[0] or j >= y.shape[0]:
        return
    if j >= i:
        return
    r = pbc_dist_cu(x[i], y[j], box)
    ret[j, i] = r
    ret[i, j] = r


def cluster(pos, box, weights=None, epsilon=1.08,
            minpts=5, gpu=0):  # these parameters are suitable for dpd with density = 3.0
    ret = np.zeros((pos.shape[0],) * 2, dtype=np.float)
    device = cuda.get_current_device()
    tpb2d = (device.WARP_SIZE,) * 2
    bpg2d = (ceil(pos.shape[0] / tpb2d[0]), ceil(pos.shape[0] / tpb2d[1]))
    with cuda.gpus[gpu]:
        pbc_pairwise_distance[bpg2d, tpb2d](pos, pos, box, ret)
    db_fitted = DBSCAN(metric='precomputed',
                       n_jobs=-1, eps=epsilon, min_samples=minpts).fit(ret, sample_weight=weights)
    # clusters = [pos[db_fitted.labels_ == _]
    #            for _ in list(set(db_fitted.labels_)) if _ != -1]
    # noises = pos[db_fitted.labels_ == -1]
    return db_fitted.labels_

