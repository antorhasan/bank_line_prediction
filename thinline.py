import cv2 
import numpy as np 
from preprocess import path_sort
from view import mean_std
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
#from


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

def data_ag(bank):
    path = path_sort('./data/exp1/')
    #path = path[0:2]
    data = []
    for i in range(len(path)):
        print(i)
        img = cv2.imread('./data/exp1/'+str(path[i])+'.png',0)
        coun = 0
        for j in range(img.shape[0]):

            for k in range(img.shape[1]):

                if img[j,k] == 255 :
                    if bank == 'both':
                        data.append(k)

                    if bank == 'left' :
                        if coun % 2 != 0 :
                            coun+=1
                            continue
                        data.append(k)
                        coun+=1
                    if bank == 'right' :
                        if coun % 2 == 0 :
                            coun+=1
                            continue
                        data.append(k)
                        coun+=1
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

def full_normalize(data,folder):
    '''normalize the data and save meann,std and constants for rescaling'''
    #data = data_ag()

    data = np.asarray(data)
    #data = np.where(data>505,data-505,505-data)
    
    mean, std = mean_std(data,folder)
    #print(data[0:10])
    data = np.asarray(data)
    data = (data - mean)/std
    #print(data[0:10])
    new_data = change_range(data,folder)
    #print(new_data[0:10])
    return new_data

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_data(mode):
    '''read each image and append all x coordinates of 255 valued pixels.
    each cloum now has time series values for a specific pixel.
    appending each colum to a list and turning that into array gives us
    an array with each row corresponding to times series values'''

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

    
    writer = tf.io.TFRecordWriter('./data/record/full_thin/'+mode+'.tfrecords')
    for i in range(len(struc)):
        '''change value to divide between train, val, test'''
        if mode == 'test' :
            start = 28 
            finish = 32
        if mode == 'train' :
            start = 0
            finish = 28
        if mode == 'val' :
            start = 2
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

def write_data_f(folder, mode,name):
    path = path_sort('./data/exp1/')
    #path = path[0:4]
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
    struc = np.asarray(struc)
    print(struc.dtype)
    struc = struc.astype('float32')
    coun = 0

    lef_mean = np.load('./data/numpy_arrays/left/mean.npy')
    lef_std = np.load('./data/numpy_arrays/left/std.npy')
    lef_a = np.load('./data/numpy_arrays/left/a.npy')
    lef_b = np.load('./data/numpy_arrays/left/b.npy')
    rg_mean = np.load('./data/numpy_arrays/right/mean.npy')
    rg_std = np.load('./data/numpy_arrays/right/std.npy')
    rg_a = np.load('./data/numpy_arrays/right/a.npy')
    rg_b = np.load('./data/numpy_arrays/right/b.npy')
    print(lef_mean,rg_mean)
    print(struc)
    print(struc.shape)
    for i in range(struc.shape[0]):
        if coun % 2 == 0 :
            var = (struc[i]-lef_mean)/lef_std
            struc[i] = (lef_a*var) + lef_b
        elif coun % 2 != 0 :
            var = (struc[i]-rg_mean)/rg_std
            struc[i] = (rg_a*var) + rg_b
        coun += 1

    print(struc[0:10,:])
    writer = tf.io.TFRecordWriter('./data/record/'+folder+'/'+name+'.tfrecords')
    for i in range(len(struc)):
        '''change value to divide between train, val, test'''
        if mode == 'test' :
            start = 3 
            finish = 32
        if mode == 'train' :
            start = 0
            finish = 28
        if mode == 'val' :
            start = 1
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

    image_y = tf.decode_raw(parsed_features["image_y"],  tf.float32)

    image_y = tf.cast(image_y,dtype=tf.float32)

    #mean = np.load('./data/numpy_arrays/thin_line/mean.npy')
    #std = np.load('./data/numpy_arrays/thin_line/std.npy')
    #a = np.load('./data/numpy_arrays/thin_line/a.npy')
    #b = np.load('./data/numpy_arrays/thin_line/b.npy')

    #image_y = (image_y-mean)/std

    #image_y = (image_y*a) + b  



    return image_y

def read_tfrecord():
    tf.enable_eager_execution()

    dataset = tf.data.TFRecordDataset('./data/record/thin/train.tfrecords')
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
    dataset = dataset.batch(14)
    mean = np.load('./data/numpy_arrays/thin_line/mean.npy')
    std = np.load('./data/numpy_arrays/thin_line/std.npy')
    a = np.load('./data/numpy_arrays/thin_line/a.npy')
    b = np.load('./data/numpy_arrays/thin_line/b.npy')
    coun = 1
    for i in dataset:
        print(coun)
        '''we have to skip 2 samples startin at 27 at an interval of 28'''
        """ if (coun-27)%28 == 0 or (coun-28)%28 == 0 :
            coun +=1
            continue """
        
        print(i)
        #print(i[:,0:2,:], i[:,2:3,:])
        i = (i-mean)/std
        i = (i*a) + b  
        print(i)
        #i = (i - b) / a 
        i = (i * std) + mean

        print(i)
        if coun > 30 :
            break
        coun += 1

def read_tfrecord_norm():
    tf.enable_eager_execution()

    dataset = tf.data.TFRecordDataset('./data/record/normal_dis/val28.tfrecords')
    dataset = dataset.map(_parse_function)
    #dataset = dataset.window(size=3, shift=1, stride=1,drop_remainder=True).flat_map(lambda x: x.batch(3))
    '''behold really efficient pipeline
    firs line-creates a dataset of tensors of size 'batch size'
    second - maps each tensor to a dataset and so converts each into a dataset ,resulting
    in a dataset of datasets
    third - each dataset from the dataset is filtered out with a sliding window and flattened
    fourth - all the tensors are turned into batches : need more explanation '''
    dataset = dataset.window(size=28, shift=28, stride=1,drop_remainder=False).flat_map(lambda x: x.batch(28))
    dataset = dataset.map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    dataset = dataset.flat_map(lambda x: x.window(size=28, shift=1, stride=1,drop_remainder=True))
    dataset = dataset.flat_map(lambda x: x.batch(28))
    dataset = dataset.batch(4)
    lef_mean = np.load('./data/numpy_arrays/left/mean.npy')
    lef_std = np.load('./data/numpy_arrays/left/std.npy')
    lef_a = np.load('./data/numpy_arrays/left/a.npy')
    lef_b = np.load('./data/numpy_arrays/left/b.npy')
    rg_mean = np.load('./data/numpy_arrays/right/mean.npy')
    rg_std = np.load('./data/numpy_arrays/right/std.npy')
    rg_a = np.load('./data/numpy_arrays/right/a.npy')
    rg_b = np.load('./data/numpy_arrays/right/b.npy')

    coun = 0

    #for i in dataset:
        #print(i)
    """ for i in dataset:
        print(coun)
        for j in range(4):
            
            if j%2==0 :
                stuff = (i[j,:,:] - lef_b) / lef_a
                stuff = (stuff*lef_std) + lef_mean
            if j%2 !=0 :
                stuff = (i[j,:,:] - rg_b) / rg_a
                stuff = (stuff*rg_std) + rg_mean
            #print(stuff.shape)
            print(stuff)
        if coun > 1 :
            break
        coun += 1 """

    '''this is for rescaling ouput'''
    for i in dataset:
        print(i)
        for j in range(4):
            print(coun)
            if j%4==0 or j%4==1:
                stuff = (i[j,:,:] - lef_b) / lef_a
                stuff = (stuff*lef_std) + lef_mean
            
            if (j-2)%4 == 0 or (j-2)%4 == 1:
                stuff = (i[j,:,:] - rg_b) / rg_a
                stuff = (stuff*rg_std) + rg_mean
            #print(stuff.shape)
            print(stuff)
        if coun > 1 :
            break
        coun += 1


#write_data_f('normal_dis', 'train')
#write_data_f('normal_dis', 'val', 'val28')
#write_data_f('normal_dis', 'test', 'test28')

read_tfrecord_norm()

#trip_thin_line()
#single_pix()
#standar_height()
#full_normalize(data)
#write_data('train')
#write_data('val')
#write_data('test')
#read_tfrecord()

""" arr_left = data_ag('left')
arr_right = data_ag('right')
print(arr_left[0:20],arr_right[0:20])
#print(arr[0:40])
out_right = full_normalize(arr_right, 'right')
out_left = full_normalize(arr_left, 'left')

concat = np.concatenate((out_left,out_right), axis=0)
plt.hist(concat,bins=200)
plt.show()
 mean = np.load('./data/numpy_arrays/right/mean.npy')
std = np.load('./data/numpy_arrays/right/std.npy')
a = np.load('./data/numpy_arrays/right/a.npy')
b = np.load('./data/numpy_arrays/right/b.npy')
arr = (arr-mean)
plt.hist(arr,bins=100)
plt.show()
arr = arr/std
plt.hist(arr,bins=100)
plt.show()
arr = (arr*a) 
plt.hist(arr,bins=100)
plt.show()
arr = arr + b
plt.hist(arr,bins=100)
plt.show()"""