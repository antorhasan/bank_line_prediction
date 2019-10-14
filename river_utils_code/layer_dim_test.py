
import numpy as np
import tensorflow as tf
#import cv2
#import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

ops.reset_default_graph()
W1 = tf.get_variable('weights1',(7,7,1,1),initializer=tf.contrib.layers.variance_scaling_initializer())
W2 = tf.get_variable('weights2',(5,5,1,1),initializer=tf.contrib.layers.variance_scaling_initializer())
W3 = tf.get_variable('weights3',(3,3,1,1),initializer=tf.contrib.layers.variance_scaling_initializer())
W4 = tf.get_variable('weights4',(1,1,1,1),initializer=tf.contrib.layers.variance_scaling_initializer())


Z = tf.random_normal(shape=(1,256,256,1))

#prime_conv1 = tf.nn.conv2d(Z,W1,strides=[1,2,2,1], padding="VALID", name="prime_conv")
prime_conv2 = tf.nn.conv2d(Z,W3,strides=[1,2,2,1], padding="VALID", name="prime_conv")
prime_conv3 = tf.nn.conv2d(prime_conv2,W3,strides=[1,2,2,1], padding="VALID", name="prime_conv")
prime_conv4 = tf.nn.conv2d(prime_conv3,W3,strides=[1,2,2,1], padding="VALID", name="prime_conv")
prime_conv5 = tf.nn.conv2d(prime_conv4,W3,strides=[1,2,2,1], padding="VALID", name="prime_conv")
prime_conv6 = tf.nn.conv2d(prime_conv5,W3,strides=[1,2,2,1], padding="VALID", name="prime_conv")
prime_conv7 = tf.nn.conv2d(prime_conv6,W3,strides=[1,2,2,1], padding="VALID", name="prime_conv")
prime_conv8 = tf.nn.conv2d(prime_conv7,W3,strides=[1,1,1,1], padding="VALID", name="prime_conv")


#ops.reset_default_graph()  
sess = tf.Session()

sess.run(tf.global_variables_initializer())
#last = sess.run(prime_conv7)
#print(prime_conv1.shape)
print(prime_conv2.shape)
print(prime_conv3.shape)
print(prime_conv4.shape)
print(prime_conv5.shape)
print(prime_conv6.shape)
print(prime_conv7.shape)
print(prime_conv8.shape)

print(prime_conv8.get_shape().as_list()[1])

sess.close()
#3,5,12,26,56/5,54/3,116/5,238

ops.reset_default_graph()
W1 = tf.get_variable('weights4',(3,3,1,1),initializer=tf.contrib.layers.variance_scaling_initializer())
#W2 = tf.get_variable('weights5',(5,5,1,1),initializer=tf.contrib.layers.variance_scaling_initializer())
#W3 = tf.get_variable('weights6',(3,3,1,1),initializer=tf.contrib.layers.variance_scaling_initializer())
Z1 = tf.random_normal(shape=(1,2,2,1))

up_pixel = tf.image.resize_nearest_neighbor(Z1, size=(4,4), name="nearest_pixel_up")
prime_conv = tf.nn.conv2d_transpose(up_pixel, W1, output_shape= (1,6,8,1), strides=[1,1,1,1], padding="VALID", name="prime_conv")

#Z = tf.random_normal(shape=(1,512,512,1))

#prime_conv1 = tf.nn.conv2d(Z,W1,strides=[1,2,2,1], padding="VALID", name="prime_conv")


#ops.reset_default_graph()  
sess = tf.Session()

sess.run(tf.global_variables_initializer())
#last = sess.run(prime_conv7)
print(prime_conv.shape)
#print(sess.run(prime_conv))


sess.close()

ops.reset_default_graph()

A = tf.constant([[[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]]]])
B = tf.random_normal(shape=(1,2,2,1))

up_a = tf.image.resize_nearest_neighbor(B,size=(2,2))


sess=tf.Session()
print(sess.run(B))
print(sess.run(up_a))
print(up_a.shape)
sess.close()
