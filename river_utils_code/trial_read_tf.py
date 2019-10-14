
from random import shuffle
import glob
import sys
import cv2
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join

def _parse_function(example_proto):
    features = {
#                'height': tf.FixedLenFeature([], tf.int64),
#                'width': tf.FixedLenFeature([], tf.int64),
#                'depth': tf.FixedLenFeature([], tf.int64),
                "image_y": tf.FixedLenFeature((), tf.string ),
                "image_m": tf.FixedLenFeature((), tf.string ),
                "image_x": tf.FixedLenFeature((), tf.string )
                }

    parsed_features = tf.parse_single_example(example_proto, features)
    
    image_y = tf.decode_raw(parsed_features["image_y"],  tf.float64)
    image_m = tf.decode_raw(parsed_features["image_m"],  tf.float64)
    image_x = tf.decode_raw(parsed_features["image_x"],  tf.float64)
#     image_y = np.fromstring(parsed_features["image_y"],dtype=float)
#     image_m = np.fromstring(parsed_features["image_m"],dtype=float)
#     image_x = np.fromstring(parsed_features["image_x"],dtype=float)
   
    #image_y = tf.reshape(image_y, [512,512,2])
    #height = tf.cast(parsed_features['height'], tf.int32)
    #width = tf.cast(parsed_features['width'], tf.int32)
    #depth = tf.cast(parsed_features['depth'], tf.int32)
#     image_y = tf.cast(image_y, tf.float32)
#     image_m = tf.cast(image_m, tf.float32)
#     image_x = tf.cast(image_x, tf.float32)
    
#     image_y = np.reshape(image_y,(512,512,1))
#     image_m = np.reshape(image_m,(512,512,1))
#     image_x = np.reshape(image_x,(512,512,1))
    
    
    image_y = tf.reshape(image_y, [512,512,1])
    image_m = tf.reshape(image_m, [512,512,1])
    image_x = tf.reshape(image_x, [512,512,1])

    #return parsed_features["image_y"],parsed_features["image_m"],parsed_features["image_x"]
    return image_y,image_m,image_x

filenames = "F:/D/Papers/train_mini.tfrecords"
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)
#dataset = dataset.shuffle(10)
dataset = dataset.batch(1)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

sess = tf.Session()
sess.run(iterator.initializer)
for i in range(10):
    print(sess.run([next_element]))
sess.close()

# dataset = tf.data.Dataset.from_tensor_slices((data_x, data_m, data_y))
# dataset = dataset.repeat(num_epochs)
# dataset = dataset.shuffle(9984)
# dataset = dataset.batch(mini_size)
# iterator = dataset.make_initializable_iterator()
# next_element = iterator.get_next()
