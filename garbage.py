from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf

print(np.load('./data/numpy_arrays/mean.npy'))

""" data = np.array([[[3, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]],
                 [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]]])

#scaler = StandardScaler().fit(data)

#print(scaler.mean_)
data = tf.cast(data,dtype=tf.float32)
mean, variance = tf.nn.moments(data, axes=[0])

sess = tf.Session()
print(sess.run(mean), sess.run(variance))


sess.close() """
 