import numpy as np 
import tensorflow as tf 
from os import listdir
from os.path import isfile, join
import cv2
import rasterio

def path_sort(path):
    '''gets a path as input and returns a list of sorted filenames'''
    image_path = path
    img_lis = [f for f in listdir(image_path) if isfile(join(image_path, f))]

    '''sort the data'''
    for i in range(len(img_lis)):
        img_lis[i] = int(img_lis[i].split('.')[0])
    img_lis.sort()

    return img_lis

def filenames_to_array(image_name_list, image_path):
    '''crops 256*768 size images and returns the aggregated numpy array'''
    
    data = []
    
    for i in range(len(image_name_list)):

        img_y = cv2.imread(image_path + str(image_name_list[i])+'.png')
        for j in range(3):
            crop_img = img_y[:,256*j:256*(j+1)]
            data.append(crop_img)

        print(i)
    data = np.asarray(data)

    return data

def array_mean_var(np_array):

    data = tf.cast(np_array,dtype=tf.float32)
    mean, variance = tf.nn.moments(data, axes=[0])
    #shapem = tf.shape(mean)
    #shapev = tf.shape(variance)

    sess = tf.Session()
    mean = sess.run(mean)
    variance = sess.run(variance)
    #print(sess.run(shapem),sess.run(shapev))
    sess.close()
    return mean, variance
    
"""     

img_list = path_sort('./data/crop1/')
array = filenames_to_array(img_list, './data/crop1/')
mean , variance = array_mean_var(array)
print(mean.shape)
np.save('./data/numpy_arrays/mean', mean) """
#ar = np.load('./data/mean.np')
#print(ar)
#print(ar.shape)
#print(mean.shape)
def save_png():


if __name__ == "__main__":

    pass