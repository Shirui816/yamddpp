from numba import cuda
import numpy as np
import numba as nb


@cuda.jit("void(int64, int64[:], int64[:])", device=True)
def unravel_index_f_cu(i, dim, ret):  # unravel index in Fortran way.
    for k in range(dim.shape[0]):
        ret[k] = int(i % dim[k])
        i = (i - ret[k]) / dim[k]


def unravel_indices_f(dim, gpu=0):
    n_dim = dim.shape[0]
    n = np.multiply.reduce(dim)
    ret = np.zeros((n, n_dim), dtype=np.int64)
    dim = np.asarray(dim, dtype=np.int64)

    @cuda.jit("void(int64[:], int64, int64[:,:])")
    def unravel_indices_f_cu(_dim, _n, _ret):
        i = cuda.grid(1)
        tmp = cuda.local.array(n_dim, nb.int64)  # n_dim must be constant in this def.
        # private array for every thread.
        if i < n:
            unravel_index_f_cu(i, _dim, tmp)
            for j in range(tmp.shape[0]):
                _ret[i, j] = tmp[j]

    with cuda.gpus[gpu]:
        device = cuda.get_current_device()
        tpb = device.WARP_SIZE
        bpg = int(np.ceil(n / tpb))
        unravel_indices_f_cu[bpg, tpb](dim, n, ret)
    return ret


@cuda.jit  # x can be array of any dimension
def _cu_kernel(x, r, r_bin, r_max2, ret, cter):
    i = cuda.grid(1)
    if i >= x.shape[0]:
        return
    tmp = 0
    j = i
    for k in range(r.shape[0]):  # r.shape == (n-dim, n-coordinates in each dim)
        idx = int(j % r.shape[1])
        j = (j - idx) / r.shape[1]
        tmp += r[k, idx] ** 2
    if tmp < r_max2:
        jdx = int(tmp ** 0.5 / r_bin)
        cuda.atomic.add(ret, jdx, x[i])
        cuda.atomic.add(cter, jdx, 1)


# TODO: complex version is not correct
@cuda.jit  # x, y can be arrays of any dimension
def _cu_kernel_complex(x, y, r, r_bin, r_max2, ret_real, ret_imag, cter):
    i = cuda.grid(1)
    if i >= x.shape[0]:
        return
    tmp = 0
    j = i
    for k in range(r.shape[0]):  # r.shape == (n-dim, n-elements on each dim)
        idx = int(j % r.shape[1])
        j = (j - idx) / r.shape[1]
        tmp += r[k, idx] ** 2
    if tmp < r_max2:
        jdx = int(tmp ** 0.5 / r_bin)
        cuda.atomic.add(ret_real, jdx, x[i])  # currently cuda.atomic.add does not support np.complex
        cuda.atomic.add(ret_imag, jdx, y[i])
        cuda.atomic.add(cter, jdx, 1)


def hist_vec_by_r_cu(x, r, r_bin, r_max, gpu=0):
    r"""Summing vector based function to modulus based function.
    $f(r) := \int F(\bm{r})\delta(r-|\bm{r}|)\mathrm{d}\bm{r} / \int \delta(r-|\bm{r}|)\mathrm{d}\bm{r}$
    :param x: np.ndarray, input
    :param r: np.ndarray[ndim=2], x[dim_1, dim_2, ..., dim_n] ~ (r[1, dim_1(i)], r[2, dim_2(j), ...)
    :param r_bin: double, bin size of r
    :param r_max: double, max of r
    :param gpu: int gpu number
    :return: np.ndarray, averaged $F(x, y, ...) -> f(\sqrt{x^2+y^2+...})$
    """
    r_max2 = r_max ** 2
    ret = np.zeros(int(r_max / r_bin) + 1, dtype=np.float)
    cter = np.zeros(ret.shape, dtype=np.uint32)
    x = x.ravel(order='F')
    y = x.imag
    x = x.real
    with cuda.gpus[gpu]:
        device = cuda.get_current_device()
        tpb = device.WARP_SIZE
        bpg = int(np.ceil(x.shape[0] / tpb))
        if np.issubdtype(x.dtype, np.complex):
            ret_imag = np.zeros(int(r_max / r_bin) + 1, dtype=np.float)
            _cu_kernel_complex[bpg, tpb](x, y, r, r_bin, r_max2, ret, ret_imag, cter)
            ret = ret + ret_imag * 1j
        else:
            _cu_kernel[bpg, tpb](x, r, r_bin, r_max2, ret, cter)
    cter[cter == 0] = 1
    return ret / cter
