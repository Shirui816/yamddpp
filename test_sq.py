from lib.Scattering import scatter_xy
import numpy as np

x = np.random.random((100,3))
y = np.random.random((100,3))

#np.savetxt('x.txt', x)
#np.savetxt('y.txt', y)

#x = np.loadtxt('x.txt')
#y = np.loadtxt('y.txt')

ret = scatter_xy(x, y, x_range=[(0,1.)]*3, r_cut=0.3, q_bin=1, q_max=20, zero_padding=1, expand=0, use_gpu=3)

ret2 = scatter_xy(x, y, x_range=[(0,1.)]*3, r_cut=0.3, q_bin=1, q_max=20, zero_padding=1, expand=0, use_gpu=False)
print(ret, ret2)