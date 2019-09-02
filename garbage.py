import numpy as np
mean = np.load('./data/numpy_arrays/thin_line/mean.npy')
std = np.load('./data/numpy_arrays/thin_line/std.npy')
a = np.load('./data/numpy_arrays/thin_line/a.npy')
b = np.load('./data/numpy_arrays/thin_line/b.npy')

print(mean, std, a, b)