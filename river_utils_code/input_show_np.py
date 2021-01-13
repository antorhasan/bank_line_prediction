
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from os import listdir
from os.path import isfile, join



A = tf.random_uniform((2,2,2,1))



trainX = "E:/antor/mini_Y/"
trainX_list = [f for f in listdir(trainX) if isfile(join(trainX,f))]
trainX_list = trainX_list[0:10880]
images = []
counter = 0

for i in trainX_list:
    img = np.asarray(cv2.imread(trainX + i,cv2.IMREAD_GRAYSCALE))
    images.append(img)
    print(counter)
    counter+=1

images = np.asarray(images)
last_y = np.reshape(images, (10880,512,512,1))
#print(last_y[0])
#converted_y = tf.convert_to_tensor(last_y,dtype=tf.float32)
#reshaped_y = tf.reshape(converted_y,shape=(11016,512,512,1))
#divided_y = tf.divide(reshaped_y,tf.constant(255.0))
divided_y = last_y/255
#print(divided_y[0])
#shuffled_y = tf.random_shuffle(divided_y)
np.random.shuffle(divided_y)
#print(divided_y[0])
#y_train,y_val = tf.split(shuffled_y,[10000,1016], axis=0)
y_train,y_val = np.split(divided_y,[9984])


trainX = "E:/antor/mini_M/"
trainX_list = [f for f in listdir(trainX) if isfile(join(trainX,f))]
trainX_list =trainX_list[0:10880]
images = []
counter = 0

for i in trainX_list:
    img = np.asarray(cv2.imread(trainX + i,cv2.IMREAD_GRAYSCALE))
    images.append(img)
    print(counter)
    counter+=1
images = np.asarray(images)
last_m = np.reshape(images,(10880,512,512,1))

#converted_m = tf.convert_to_tensor(last_m,dtype=tf.float32)
#reshaped_m = tf.reshape(converted_m,shape=(11016,512,512,1))

#final_mask = tf.where(tf.not_equal(reshaped_m, tf.constant(255.0)),tf.zeros_like(reshaped_m),reshaped_m)
last_m = np.where(last_m==255,1,0)
#final_bm = tf.where(tf.not_equal(final_mask, tf.constant(0.0)),tf.ones_like(final_mask),final_mask)
#shuffled_m = tf.random_shuffle(final_bm)
np.random.shuffle(last_m)
last_m = last_m.astype(float)
print(last_m.dtype)
#m_train,m_val = tf.split(shuffled_m,[10000,1016], axis=0)
m_train,m_val = np.split(last_m,[9984])

#x_final = tf.multiply(shuffled_y,shuffled_m)
x_final = np.multiply(divided_y,last_m)
#x_train,x_val = tf.split(x_final,[10000,1016], axis=0)
x_train,x_val = np.split(x_final,[9984])

print(x_train.shape,x_val.shape,y_train.shape,y_val.shape,m_train.shape,m_val.shape)
#print(y_train[0])
#print(m_train[0])
#print(x_train[0])
#print(m_train.dtype)
#m_train = m_train.astype(float)
#print(m_train.dtype)
#print(m_val.dtype)
#print(x_train.dtype)


sess = tf.Session()
#B = A.get_shape()
#C = A.set_shape([2,2,3,1])
#B[2] =3
G = tf.TensorShape([None,2,2,1])
print(G)
#H = tf.random_uniform()
sess.close()


trainX = "F:/D/Papers/mini_X/"
trainX_list = [f for f in listdir(trainX) if isfile(join(trainX,f))]
trainX_list =trainX_list[0:1]
a = np.empty(0)
counter = 0

for i in trainX_list:
    img = cv2.imread(trainX + i,cv2.IMREAD_GRAYSCALE)
    img_flat = np.ndarray.flatten(img)
    a = np.append(a, img_flat)
    counter+=1
    print(counter)
last1 = np.reshape(a, (1,512,512))
#last = last/255
print(last1)
#converted = tf.convert_to_tensor(last,dtype=tf.float32)
#reshaped = tf.reshape(converted,shape=(1,512,512,1))
#x_train,x_val = tf.split(reshaped,[8,2], axis=0)

trainX = "F:/D/Papers/mini_Y/"
trainX_list = [f for f in listdir(trainX) if isfile(join(trainX,f))]
trainX_list =trainX_list[0:1]
a = np.empty(0)
counter = 0

for i in trainX_list:
    img = cv2.imread(trainX + i,cv2.IMREAD_GRAYSCALE)
    img_flat = np.ndarray.flatten(img)
    a = np.append(a, img_flat)
    counter+=1
    print(counter)
last2 = np.reshape(a, (1,512,512))
print(last2)

#converted = tf.convert_to_tensor(last,dtype=tf.float32)
#reshaped = tf.reshape(converted,shape=(2,512,512,1))
#y_train,y_val = tf.split(reshaped,[8,2], axis=0)



trainX = "F:/D/Papers/mini_M/"
trainX_list = [f for f in listdir(trainX) if isfile(join(trainX,f))]
trainX_list =trainX_list[0:1]
a = np.empty(0)
counter = 0

for i in trainX_list:
    img = cv2.imread(trainX + i,cv2.IMREAD_GRAYSCALE)
    img_flat = np.ndarray.flatten(img)
    a = np.append(a, img_flat)
    counter+=1
    print(counter)
last3 = np.reshape(a, (1,512,512))
print(last3)




converted = tf.convert_to_tensor(last3,dtype=tf.float32)
final = tf.where(tf.not_equal(converted, tf.constant(255.0)),tf.zeros_like(converted),converted)
final_m = tf.where(tf.not_equal(final, tf.constant(0.0)),tf.ones_like(final),final)
x_final = tf.multiply(final_m,last1)

sess = tf.Session()
print(sess.run(final))
print(sess.run(final_m))
print(sess.run(x_final))
sess.close()


#reshaped = tf.reshape(converted,shape=(2,512,512,1))
#m_train,m_val = tf.split(reshaped,[8,2], axis=0)
