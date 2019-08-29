import tensorflow as tf
from tensorflow.keras import layers
import cv2
import numpy as np

'''read tfrecord file and view images and labels'''

def _parse_function(example_proto):

    features = {
                "image_y": tf.FixedLenFeature((), tf.string ),
                "image_m": tf.FixedLenFeature((), tf.string )
                }

    parsed_features = tf.parse_single_example(example_proto, features)

    image_y = tf.decode_raw(parsed_features["image_y"],  tf.float64)
    image_m = tf.decode_raw(parsed_features["image_m"],  tf.float64)

    image_y = tf.reshape(image_y, [256,256,6])
    image_m = tf.reshape(image_m, [256,256,1])
    image_y = tf.cast(image_y,dtype=tf.float32)
    image_m = tf.cast(image_m,dtype=tf.float32)
    #imgtf = tf.math.divide(tf.math.multiply(tf.cast(image_y,dtype=tf.int64),255),3000)
    imgtf = tf.cast(image_y, tf.uint8)
    #imgtf = tf.cast(imgtf, tf.float32)
    """ img_lab = tf.cast(image_m,dtype=tf.bool)
    img_lab = tf.math.logical_not( img_lab )
    img_lab = tf.cast(img_lab,dtype=tf.float32)
    mask = tf.concat([image_m, img_lab], 2) """

    return imgtf, image_m

dataset = tf.data.TFRecordDataset('./data/record/train_tif.tfrecords')
dataset = dataset.map(_parse_function)
#dataset = dataset.shuffle(3000)
#dataset = dataset.batch(8)
#print(dataset.output_shapes,dataset.output_types)
iterator = dataset.make_initializable_iterator()
#image, mask = iterator.get_next()

sess = tf.Session()

sess.run(iterator.initializer)
count = 0
""" while True:
    img, lab = sess.run(iterator.get_next())
    print(count)
    count += 1 """
img, lab = sess.run(iterator.get_next())
#kernel = np.ones((7,7), np.uint8)
#lab = cv2.dilate(lab, kernel, iterations=1)
print(img.shape)
#img = np.divide(np.multiply(np.int64(img), [255]), [3000])
img = img[:,:,3:6]
#img = np.divide(np.multiply(np.int64(img), [255]), [3000])
print(img)
print()
#img = np.uint8(img)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', lab)
cv2.waitKey(0)
cv2.destroyAllWindows
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', lab[0,:,:,0])
cv2.waitKey(0)
cv2.destroyAllWindows

sess.close()