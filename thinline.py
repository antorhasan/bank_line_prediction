import cv2 
import numpy as np 
from preprocess import path_sort
from view import mean_std
import tensorflow as tf
import sys


def trip_thin_line():
    '''trim the image 120 pixel on four sides'''
    path = path_sort('./data/bankline/')
    print(path)
    for i in range(len(path)):
        img = cv2.imread('./data/bankline/'+str(path[i])+'.png',0)
        img = img[121:-120,121:-120]
        cv2.imwrite('./data/bankline/'+str(path[i])+'.png',img)

def single_pix():
    '''makes sure each row of an image has only two valid pixels'''
    path = path_sort('./data/bankline/')
    #path = path[0:2]
    print(path)
    for i in range(len(path)):
        print(i)
        img = cv2.imread('./data/bankline/'+str(path[i])+'.png',0)
        coor = []
        for j in range(img.shape[0]):
            lis = []
            for k in range(img.shape[1]):
                if img[j,k] == 255 :
                    lis.append(k)
            if len(lis) != 0 :
                small = min(lis)
                big = max(lis)
                if big-small>5 :
                    coor.append(tuple([j,small]))
                    coor.append(tuple([j,big]))
        new = np.zeros((img.shape[0],img.shape[1]))
        for j in coor:
            new[j[0],j[1]] = 255
        
        cv2.imwrite('./data/exp/'+str(path[i])+'.png', new)

def standar_height():
    '''crop all images so that everyone has same number of 
    valid pixels'''
    path = path_sort('./data/exp/')
    #path = path[0:2]
    min_lis = []
    max_lis = []
    for i in range(len(path)):
        img = cv2.imread('./data/exp/'+str(path[i])+'.png',0)
        coor = np.where(img==255)
        min_lis.append(min(coor[0]))
        max_lis.append(max(coor[0]))
    print(min_lis,max_lis)
    print(max(min_lis),min(max_lis))
    for i in range(len(path)):
        img = cv2.imread('./data/exp/'+str(path[i])+'.png',0)
        img = img[max(min_lis):min(max_lis),:]
        cv2.imwrite('./data/exp1/'+str(path[i])+'.png',img)

def check_data():
    '''check if each image has same number of valid pixels'''
    path = path_sort('./data/exp1/')
    #path = path[0:2]
    for i in range(len(path)):
        img = cv2.imread('./data/exp1/'+str(path[i])+'.png',0)
        coor = np.where(img == 255)
        lis = list(zip(coor[0],coor[1]))
        for j in range(len(lis)):
            print(lis[j])
            if j==5:
                break
        print(len(coor[0]),len(coor[1]))



def check_number():
    '''check how many valid 1 pixels are in each row of a mask'''
    path = path_sort('./data/exp1/')
    path = path[0:2]

    for i in range(len(path)):
        print(i)
        img = cv2.imread('./data/exp1/'+str(path[i])+'.png',0)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows
        coor = []
        cu = 0
        for j in range(img.shape[0]):
            coun = 0
            for k in range(img.shape[1]):
                
                if img[j,k] == 255 :
                    cu += 1
                    
                    coun += 1
                if coun == 3 :
                    coor.append(j)
        print(cu)
        print(coor)

def data_ag():
    path = path_sort('./data/exp1/')
    #path = path[0:2]
    data = []
    for i in range(len(path)):
        print(i)
        img = cv2.imread('./data/exp1/'+str(path[i])+'.png',0)
        for j in range(img.shape[0]):
            for k in range(img.shape[1]):
                if img[j,k] == 255 :
                    data.append(k)
    return data


def change_range(data,folder):
    newmin = -1
    newmax = 1
    newR = newmax - newmin
    oldmin = np.amin(data)
    oldmax = np.amax(data)
    oldR = oldmax-oldmin
    a = newR / oldR
    b = newmin - ((oldmin*newR)/oldR)
    new_data = (data*a) + b
    np.save('./data/numpy_arrays/'+folder+'/a', a)
    np.save('./data/numpy_arrays/'+folder+'/b', b)
    return new_data

def full_normalize():
    '''normalize the data and save meann,std and constants for rescaling'''
    data = data_ag()
    mean, std = mean_std(data,'thin_line')
    print(data[0:10])
    data = np.asarray(data)
    data = (data - mean)/std
    print(data[0:10])
    new_data = change_range(data,'thin_line')
    print(new_data[0:10])

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_data(mode):
    path = path_sort('./data/exp1/')
    #path = path[0:3]
    full = []

    for i in range(len(path)):
        print(i)
        data = []
        img = cv2.imread('./data/exp1/'+str(path[i])+'.png',0)
        for j in range(img.shape[0]):
            for k in range(img.shape[1]):
                if img[j,k] == 255 :
                    data.append(k)
        full.append(data)

    full = np.asarray(full)
    print(np.asarray(full).shape)
    struc = []
    for i in range(full.shape[1]):
        struc.append(list(full[:,i]))
    print(struc[0:10])
    print(len(struc))

    writer = tf.io.TFRecordWriter('./data/record/thin/'+mode+'.tfrecords')
    for i in range(len(struc)):
        '''change value to divide between train, val, test'''
        if mode == 'test' :
            start = 28 
            finish = 32
        if mode == 'train' :
            start = 0
            finish = 28
        if mode == 'val' :
            start = 26
            finish = 30

        lis = struc[i][start:finish]
        if i < 10 :
            print(lis)
        for j in range(len(lis)):
            feature = {
                'image_y': _bytes_feature(lis[j].tostring())   
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()


def _parse_function(example_proto):

	features = {
				"image_y": tf.io.FixedLenFeature((), tf.string )
				}

	parsed_features = tf.io.parse_single_example(example_proto, features)

	image_y = tf.decode_raw(parsed_features["image_y"],  tf.int64)

	image_y = tf.cast(image_y,dtype=tf.float32)

	return image_y

def read_tfrecord():
    tf.enable_eager_execution()

    dataset = tf.data.TFRecordDataset('./data/record/thin/val.tfrecords')
    dataset = dataset.map(_parse_function)
    #dataset = dataset.window(size=3, shift=1, stride=1,drop_remainder=True).flat_map(lambda x: x.batch(3))
    '''behold really efficient pipeline
    firs line-creates a dataset of tensors of size 'batch size'
    second - maps each tensor to a dataset and so converts each into a dataset ,resulting
    in a dataset of datasets
    third - each dataset from the dataset is filtered out with a sliding window and flattened
    fourth - all the tensors are turned into batches : need more explanation '''
    dataset = dataset.window(size=4, shift=4, stride=1,drop_remainder=False).flat_map(lambda x: x.batch(4))
    dataset = dataset.map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    dataset = dataset.flat_map(lambda x: x.window(size=3, shift=1, stride=1,drop_remainder=True))
    dataset = dataset.flat_map(lambda x: x.batch(3))
    dataset = dataset.batch(10)

    coun = 1
    for i in dataset:
        print(coun)
        '''we have to skip 2 samples startin at 27 at an interval of 28'''
        """ if (coun-27)%28 == 0 or (coun-28)%28 == 0 :
            coun +=1
            continue """
        
        print(i)
        print(i[:,0:2,:], i[:,2:3,:])
        if coun > 10 :
            break
        coun += 1


#trip_thin_line()
#single_pix()
#standar_height()
#full_normalize()
#write_data('train')
#write_data('val')
#write_data('test')
#read_tfrecord()
