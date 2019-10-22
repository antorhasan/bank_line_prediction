import rasterio
import cv2
from rasterio.plot import show
from rasterio.plot import show_hist
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

#from preprocess import *
import sys
#from preprocess import path_sort
from utility import single_pix

class viz():
    '''read tif image and get relevant properties
    '''
    def __init__(self, path):
        '''parse input tif fime to numpy array
        Args :
            - path : .tif file path'''
        self.path = path
        data = rasterio.open(self.path)
        data = data.read()
        img = rasterio.plot.reshape_as_image(data)
        img_np = np.asarray(img)
        self.np_img = img_np

    def get_image(self,img_type,norm):
        img_np = self.np_img
        if img_type == 'rgb' :
            img = img_np[:,:,0:3]
        elif img_type == 'infra' :
            img = img_np[:,:,3:6]
        elif img_type == 'raster' :
            img = img_np[:,:]
            #print(img.shape)
        #print(img.shape)
        '''the int16 format of the input channels needs to be changed into regular int64
        format in order to broadcast properly with int64 numpy array'''
        if norm == True :
            img = np.divide(np.multiply(np.int64(img), [255]), [3000])
        elif norm == False :
            img = np.where(np.int64(img)!=0,255,0)
        
        #conversion is needed to unit8 to keep the range between 0-255
        self.img = np.uint8(img)
        #self.img_np = img_np
    
    def get_array(self):
        return np.asarray(self.img)

    def cv_view(self):
        '''view raster file's first 3 channel in opencv, input: raster object. 
        output: visualization of the first 3 channel in opencv'''

        '''opencv reads files in BGR channel format. Here this is satisfied and no 
        conversion is needed'''
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def cv_write(self, output_dir,filename):
        self.img = self.img[13:,:]
        self.img = self.img[0:2048,:]
        cv2.imwrite(output_dir + filename +'.png', self.img)



def tif_to_npaggr(path):
    
    path_list = path_sort(path)
    #path_list = path_list[0:10]
    data = []
    for i in range(len(path_list)):
        print(i)
        #print('./data/finaltif/'+str(path_list[i]))
        img = viz(path+str(path_list[i])+'.tif')

        img_full = img.get_array()
        coor_list = [516, 491, 516, 450, 396, 355, 325, 277, 310, 400]

        for j in range(len(coor_list)):
            num = coor_list[j]
            crop_img = img_full[256*j : 256*(j+1),num:num+768]

            for k in range(3):
                crop_img_256 = crop_img[:,256*k:256*(k+1)]
                #crop_img_256 = np.int64(crop_img_256)
                '''the following two lines are needed to represent data in 
                understandable image format'''
                crop_img_256 = np.divide(np.multiply(np.int64(crop_img_256), [255]), [3000]) 
                #print(crop_img_256.dtype, crop_img_256)
                crop_img_256 = np.uint8(crop_img_256)

                #print(crop_img_256)
                #print(np.float32(crop_img_256))
                data.append(crop_img_256)
    return data


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def createDataRecord_tif(out_filename, addrs_y, addrs_m):

    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(addrs_y)):
        print(i)
        #print('./data/finaltif/'+str(path_list[i]))
        img = viz('./data/finaltif/'+str(addrs_y[i])+'.tif')

        img_full = img.get_array()
        coor_list = [516, 491, 516, 450, 396, 355, 325, 277, 310, 400]

        for j in range(len(coor_list)):
            num = coor_list[j]
            crop_img = img_full[256*j : 256*(j+1),num:num+768]

            img_m = np.asarray(cv2.imread(trainM + str(addrs_m[j])+'.png',cv2.IMREAD_GRAYSCALE)) 

            for k in range(3):
                crop_img_256 = crop_img[:,256*k:256*(k+1)]
                crop_img_256 = np.divide(np.multiply(np.int64(crop_img_256), [255]), [3000]) 
                #crop_img_256 = np.uint8(crop_img_256)
                #print(crop_img_256.shape)
                #data.append(crop_img_256)
                imgm = img_m[:,256*k:256*(k+1)]
                imgm = np.reshape(imgm,(256,256,1))
                last_m = imgm/255
                kernel = np.ones((3,3), np.uint8)
                last_m = cv2.dilate(last_m, kernel, iterations=1)
                
                feature = {
                    'image_y': _bytes_feature(crop_img_256.tostring()),
                    'image_m': _bytes_feature(last_m.tostring())
                    #'image_x': _bytes_feature(last_x.tostring())     
                }
        
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                
                writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()
        
def write_data():
    path_list = path_sort('./data/finaltif/')
    trainM = "./data/infralabel/"
    trainM_list = path_sort(trainM)
    print(len(path_list),len(trainM_list))
    train_M = trainM_list[0:280]
    val_M = trainM_list[280:300]
    path_list_train = path_list[0:28]
    path_list_val = path_list[28:30]

    createDataRecord_tif("./data/record/train_tif.tfrecords", path_list_train, train_M)
    createDataRecord_tif("./data/record/val_tif.tfrecords", path_list_val, val_M)


def check_zero():
    path = path_sort('./data/raster_o/raster/')
    for i in range(len(path)):
        print(i)
        data  = viz('./data/raster_o/raster/'+str(path[i])+'.tif')
        arr = data.get_array()
        arr = arr.reshape((2638,1403))
        coun = 0
        for j in range(arr.shape[0]):
            for k in range(arr.shape[1]):
                if arr[j,k] != 0 :
                    coun+=1
                    print(i,'start')
                    print(coun)

def create_tif_img(path,dest):
    path = path_sort(path)
    for i in range(len(path)):
        print(i)
        data  = viz('./data/raster_o/raster/'+str(path[i])+'.tif')
        arr = data.get_array()
        arr = arr.reshape((2638,1403))
        new_img = np.zeros((2638,1403),dtype=np.uint8)
        coun = 0
        for j in range(arr.shape[0]):
            coor_list = []
            for k in range(arr.shape[1]):
                if arr[j,k] == 1 :
                    coor_list.append(k)
            coor_list = np.asarray(coor_list)
            max_co = np.amax(coor_list)
            min_co = np.amin(coor_list)
            new_img[j,max_co] = 1
            new_img[j,min_co] = 1
        
        cv2.imwrite('./data/')

def check_img_dist():
    '''plot distribution of all images in a directory'''
    path = './data/img/final/'
    path_ls = [f for f in listdir(path) if isfile(join(path, f))]
    lis = []
    #path_ls = path_ls[0:2]
    for i in range(len(path_ls)):
        img = cv2.imread(path+path_ls[i],1)
        #print(img)
        img = list(img)
        #print(img)
        lis.append(img)
    #print(lis)
    lis = np.asarray(lis)
    lis = lis.flatten()
    #lis = np.log(lis)
    plt.hist(lis, bins=200)
    plt.show()

def tif_to_png_lines():
    path = './data/ras_final/'
    path_ls = [f for f in listdir(path) if isfile(join(path, f))]
    for i in range(len(path_ls)):
        print(path_ls[i])
        img = viz(path + path_ls[i])
        img.get_image('raster',norm=False)
        #img.cv_view()
        img.cv_write('./data/img/png/',path_ls[i].split('.')[0])


if __name__ == "__main__" :
    #check_img_dist()
    #img.cv_write('./data/img/png/','new')
    #single_pix('./data/img/png/', './data/img/lines/')
    pass
