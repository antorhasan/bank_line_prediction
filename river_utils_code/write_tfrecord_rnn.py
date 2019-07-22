
from random import shuffle
import glob
import sys
import cv2
import numpy as np
import tensorflow as tf
import os



def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

trainY = "/media/antor/Files/main_projects/rnn_test/*.jpg"
trainY_list = glob.glob(trainY)
only = [os.path.basename(x) for x in trainY_list]
onl = [os.path.splitext(x)[0] for x in only]

results = list(map(int, onl))
results.sort()
resul = list(map(str, results))
print(resul)





writer = tf.python_io.TFRecordWriter("/media/antor/Files/ML/Papers/train_rnn_test.tfrecords")
for i in range(len(resul)):
    print(i)
    img_y = cv2.imread("/media/antor/Files/main_projects/finally/"+resul[i]+".jpg", cv2.IMREAD_GRAYSCALE)
    img_y = cv2.fastNlMeansDenoising(img_y, h=24, templateWindowSize=7, searchWindowSize=21)

    img_y = np.asarray(img_y)
    #img_y = np.asarray(cv2.imread(addrs_y[i], cv2.IMREAD_GRAYSCALE))

    last_y = np.reshape(img_y, (256,256,1))
    last_y = last_y/255

    feature = {
        'image_y': _bytes_feature(last_y.tostring())
        #'image_x': _bytes_feature(last_x.tostring())     
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()




#train_Y = trainY_list[0:19488]
#train_M = trainM_list[0:19488]


#createDataRecord("/media/antor/Files/ML/Papers/train_rnn.tfrecords", train_Y)

    
