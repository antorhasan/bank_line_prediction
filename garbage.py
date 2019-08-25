from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf

data = np.array([[[3, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]],
                 [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]]])
""" data = tf.constant([[[[2], [1], [1]],
                    [[1], [1], [1]],
                    [[1], [1], [1]]],
                   [[[3], [1], [1]],
                    [[1], [1], [1]],
                    [[1], [1], [1]]]], dtype=tf.float32) """
#scaler = StandardScaler().fit(data)

#print(scaler.mean_)
data = tf.cast(data,dtype=tf.float32)
mean, variance = tf.nn.moments(data, axes=[0])

sess = tf.Session()
print(sess.run(mean), sess.run(variance))


sess.close()
