import rasterio
import cv2
from rasterio.plot import show
from rasterio.plot import show_hist
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.metrics import mean_absolute_error
#from preprocess import *
import sys
import os
#from preprocess import path_sort
#from utility import single_pix
import matplotlib.pyplot as plt
import fiona
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


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

def view_data_1():
    '''view the dataset where '''
    dataseti = tf.data.TFRecordDataset('./data/img/record/first_img/train_28.tfrecords')
    dataseti = dataseti.map(_parse_function_img)
    #dataset = dataset.window(size=2, shift=2, stride=1, drop_remainder=False).flat_map(lambda x: x.batch(2))
    dataseti = dataseti.window(size=28, shift=28, stride=1, drop_remainder=True).flat_map(lambda x: x.batch(28))
    dataseti = dataseti.map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    dataseti = dataseti.flat_map(lambda x: x.window(size=3, shift=3, stride=1,drop_remainder=True))
    dataseti = dataseti.flat_map(lambda x: x.batch(3))
    #dataset = dataset.shuffle(3000)
    dataseti = dataseti.batch(2)

    datasetm = tf.data.TFRecordDataset('./data/img/record/first_img/train_28.tfrecords')
    datasetm = datasetm.map(_parse_function_msk)
    #dataset = dataset.window(size=2, shift=2, stride=1, drop_remainder=False).flat_map(lambda x: x.batch(2))
    datasetm = datasetm.window(size=28, shift=28, stride=1, drop_remainder=True).flat_map(lambda x: x.batch(28))
    datasetm = datasetm.map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    datasetm = datasetm.flat_map(lambda x: x.window(size=3, shift=3, stride=1,drop_remainder=True))
    datasetm = datasetm.flat_map(lambda x: x.batch(3))
    #dataset = dataset.shuffle(3000)
    datasetm = datasetm.batch(2)

    for img, msk in zip(dataseti, datasetm):
        img1 = img[0,0,:,:,:]
        img2 = img[0,1,:,:,:]
        img1 = img1*255
        img2 = img2*255

        msk1 = msk[0,2,:]

        msk1 = (((msk1-b)/a) * std ) + mean

        msk = np.zeros((256,256))
        for i in range(msk.shape[0]):
            for j in range(msk.shape[1]):
                if j == int(msk1[i]):
                    msk[i,j] = 255

        img1 = np.asarray(img1,dtype=np.uint8)
        img2 = np.asarray(img2,dtype=np.uint8)
        msk = np.asarray(msk,dtype=np.uint8)
        
        cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
        cv2.imshow('image1',img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
        cv2.imshow('image1',img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
        cv2.imshow('image1',msk)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def apply_signal_denoising():
    from scipy.signal import savgol_filter
    img = cv2.imread('./data/output/10_18.png',1)

    height = img.shape[0]
    width = img.shape[1]
    """ cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
    cv2.imshow('image1',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(asd) """
    arr_0 = []
    arr_1 = []
    coun_in = 0
    for i in range(height):
        for j in range(width):
            #print(img[i,j,2])
            
            if img[i,j,0] == 0 and img[i,j,1] == 0 and img[i,j,2] == 255 :
                if coun_in == 0 :
                    arr_0.append(j)
                    coun_in += 1
                else:
                    arr_1.append(j)

        coun_in = 0
    
    print(len(arr_0))
    print(len(arr_1))
    window = 99
    poly = 2
    arr_0 = savgol_filter(arr_0, window, poly)
    arr_1 = savgol_filter(arr_1, window, poly)

    #print(arr_1)
    #print(asd)
    for i in range(height):
        for j in range(width):
            #print(int(arr_0[i]))
            if j == int(arr_0[i]) or j == int(arr_1[i]) :
                #print(j)
                img[i,j,0] = 255
                

    cv2.imwrite('./data/denoised.png',img)
    #print(img.shape)
    #print(asd)
    #height = img.()

def thinning(img):
    img = img[3:-3,3:-3]
    #img = np.where(img==255,1,255)
    img = 255-img
    img = np.where(img<10,1,img)
    img = np.asarray(img, dtype=np.uint8)
    kernel = np.ones((3,3),np.uint8)
    img = cv2.erode(img,kernel,iterations = 1)
    img = np.where(img!=1,255,img)



    return img

def custom_range(img):
    org_vec = []
    for i in range(628,1258,1):
        for j in range(img.shape[1]):
            if img[i,j] == 255:
                org_vec.append(j)
                break

    for i in range(2794,3970,1):
        for j in range(img.shape[1]):
            if img[i,j] == 255:
                org_vec.append(j)
                break
    return org_vec

def cegis():
    img_o = cv2.imread('./data/cegis/CEGIS_bank_existing_01.jpg',0)
    img_p = cv2.imread('./data/cegis/CEGIS_bank_predicted_01.jpg',0)
    
    img_o = thinning(img_o)
    img_p = thinning(img_p)

    org_vec = custom_range(img_o)
    pre_vec = custom_range(img_p)
    #print(img.shape)
    
    print(len(org_vec),len(pre_vec))
    print(org_vec)
    print(pre_vec)
    error = mean_absolute_error(org_vec, pre_vec)
    print(error)

def mean_tensor(elemt_rmv):
    '''get mean tensor of the images'''
    rgb_path = './data/img/final_rgb/'
    infra_path = './data/img/infra/'

    img_list = [f for f in listdir(rgb_path) if isfile(join(rgb_path, f))]
    img_list = [int(f.split('.')[0]) for f in img_list]
    img_list.sort()
    img_list = [str(f) for f in img_list]
    #img_list.remove('201801.png')
    for i in range(elemt_rmv):
        img_list.pop()
    
    print('calculating mean from ......')
    print(img_list)

    img = cv2.imread(rgb_path+"198801.png",1)
    mean_img = np.zeros((img.shape[0],img.shape[1],6))

    print('frequency of null values for rgb or infra ....')
    for i in range(len(img_list)):
        rgb_img = cv2.imread(rgb_path+img_list[i]+'.png',1)
        #print(rgb_img.shape)

        infra_img = cv2.imread(infra_path+img_list[i]+'.png',1)
        #print(infra_img.shape)

        comb_img = np.concatenate((rgb_img,infra_img),axis = 2)
        mean_img = mean_img + (comb_img/len(img_list))

        null_list = np.argwhere(infra_img[:,:,1]==0)
        print(img_list[i] + '  ' + str(len(null_list)))

    print("shape of mean tensor ", mean_img.shape)
    ###save mean tensor 
    np.save('./data/mean_img/mean_tensor.npy', mean_img)
    norm_mean_img = mean_img/255
    ###save normalized mean tensor
    np.save('./data/mean_img/norm_mean_img.npy', norm_mean_img)
    ###save mean image in rgb to view
    cv2.imwrite('./data/mean_img/mean_img.png',mean_img[:,:,0:3])

    """ mean_line = np.zeros((745,6))
    print(norm_mean_img.shape[0])
    for i in range(norm_mean_img.shape[0]):
        mean_line = mean_line + (norm_mean_img[i,:,:]/norm_mean_img.shape[0])
    
    np.save('./data/mean_img/mean_line.npy', mean_line) """

import rasterio.mask    
from rasterio.features import sieve

def wrt_temp_blank_tif():
    '''takes a reference tif to extract crs and transform info and
    wrtie a blank tif with the same geo properties'''
    img = rasterio.open("./data/img/finaltif/198801.tif")
    temp_arr = np.full((img.height,img.width), 255)
    temp_arr = np.asarray(temp_arr, dtype=np.uint8)

    with rasterio.open(
        './data/img/temp.tif',
        'w',
        driver='GTiff',
        height=img.height,
        width=img.width,
        count=1,
        dtype=temp_arr.dtype,
        crs=img.crs,
        transform=img.transform,
    ) as dst:
        dst.write(temp_arr, 1)


def wrt_bin_mask(img, file_id):
    '''use 255 filled georeferenced tif to write binary mask
    input : output mask from rasterio.mask.mask function
    output : 2d binary mask
    '''
    offset = 385
    img = rasterio.plot.reshape_as_image(img)
    img = np.asarray(img, dtype = np.uint8)

    img = img[:,offset:offset+745]
    print("writing binary mask.......")
    cv2.imwrite(os.path.join('./data/img/png/'+file_id+'.png'), img)

    return img

def save_img_from_tif(tif_path,img_type,norm,file_id):
    '''write rgb or ifra channel data as image data from tif file'''
    offset = 385
    jan_tif_path = tif_path
    tif_img = viz(jan_tif_path)
    tif_img.get_image(img_type=img_type,norm=norm)
    #tif_img.cv_view()
    tif_sav = tif_img.get_array()
    tif_sav = tif_sav[:,offset:offset+745,:]
    print('writing colored image file.......')
    if img_type == 'rgb':
        cv2.imwrite(os.path.join('./data/img/final_rgb/'+file_id+'.png'), tif_sav)
    elif img_type == 'infra':
        cv2.imwrite(os.path.join('./data/img/infra/'+file_id+'.png'), tif_sav)
    return tif_sav

def mask_to_bnk_list(img):
    '''generate left and right bankline list from a binary mask'''
    left_lis = [0]*img.shape[0]
    right_lis = [0]*img.shape[0]

    for i in range(img.shape[0]):
        low = 0
        high = img.shape[1] - 1 

        left_flag = True
        right_flag = True

        while low < high and (left_flag==True or right_flag==True):
            if left_flag :
                if img[i,low] == 0 :
                    low += 1
                else :
                    left_lis[i] = low
                    left_flag = False

            if right_flag :
                if img[i,high] == 0 :
                    high -= 1
                else :
                    right_lis[i] = high
                    right_flag = False

    print('''generated left and right bankline list........''')
    return left_lis, right_lis

def shp_mask_tif(shp_path, tif_path):
    file_id = tif_path.split('/')[-1].split('.')[0]
    print(file_id)
    #print(asd)

    with fiona.open(shp_path, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    temp_tif_path = os.path.join('./data/img/temp.tif')
    with rasterio.open(temp_tif_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes)
    
    img = wrt_bin_mask(out_image, file_id)

    left_lis, right_lis = mask_to_bnk_list(img)

    save_infra = save_img_from_tif(tif_path, 'infra',True, file_id)

    tif_img = save_img_from_tif(tif_path, 'rgb',True, file_id)

    for i in range(tif_img.shape[0]):
        if left_lis[i] != 0 and right_lis[i] != 0:
            tif_img[i,left_lis[i],0] = 255
            tif_img[i,left_lis[i],1] = 255
            tif_img[i,left_lis[i],2] = 255
            tif_img[i,right_lis[i],0] = 255
            tif_img[i,right_lis[i],1] = 255
            tif_img[i,right_lis[i],2] = 255

    print('''writing rgb with lines images ......''')
    cv2.imwrite(os.path.join('./data/img/shp_mask/'+file_id+'.png'), tif_img)

    lines_raster = np.zeros((tif_img.shape[0],tif_img.shape[1]))
    for i in range(lines_raster.shape[0]):
        lines_raster[i,left_lis[i]] = 255
        lines_raster[i,right_lis[i]] = 255
    
    lines_raster = np.asarray(lines_raster[0:2048,:], dtype=np.uint8)
    print('writing binary lines ......')
    cv2.imwrite(os.path.join('./data/img/lines/'+file_id+'.png'), lines_raster)

    left_lis = np.resize(left_lis,(len(left_lis),1))
    right_lis = np.resize(right_lis,(len(right_lis),1))
    line_lis = np.concatenate((left_lis,right_lis),axis=1)
    print('writing numpy array.....')
    np.save(os.path.join('./data/img/line_npy/'+file_id+'.npy'), line_lis)

def batch_shp_to_data(tif_path):
    '''using tif file path batch process shp and tif into 
    rgb, infra, binary_mask, bin_lines, line_numpy and rgb_with_lines'''

    tif_lis = [f for f in listdir(tif_path) if isfile(join(tif_path, f))]
    tif_lis = [int(f.split('.')[0]) for f in tif_lis]
    tif_lis.sort()
    tif_lis = [str(f) for f in tif_lis]
    print(tif_lis)

    shp_path = os.path.join('./data/img/shape_files/')
    for i in range(len(tif_lis)):
        shp_file = shp_path+tif_lis[i][0:-2]+'/'+tif_lis[i]+'.shp'
        shp_mask_tif(shp_file,tif_path+tif_lis[i]+'.tif')


def up_rgb_infra(inter_val):
    rgb_path  = os.path.join('./data/img/final_rgb/')
    infra_path = os.path.join('./data/img/infra/')

    img_list = [f for f in listdir(rgb_path) if isfile(join(rgb_path, f))]
    img_list = [int(f.split('.')[0]) for f in img_list]
    img_list.sort()
    img_list = [str(f) for f in img_list]

    mean_img = np.load(os.path.join('./data/mean_img/mean_tensor.npy'))
    print('updating......')
    for i in range(len(img_list)):
        print(img_list[i])
        img = cv2.imread(rgb_path+img_list[i]+'.png')
        img1 = np.resize(np.where(img[:,:,0]==0,mean_img[:,:,0],img[:,:,0]),(img.shape[0],img.shape[1],1))
        img2 = np.resize(np.where(img[:,:,1]==0,mean_img[:,:,1],img[:,:,1]),(img.shape[0],img.shape[1],1))
        img3 = np.resize(np.where(img[:,:,2]==0,mean_img[:,:,2],img[:,:,2]),(img.shape[0],img.shape[1],1))
        img = np.concatenate((img1,img2,img3), axis=2)
        cv2.imwrite(os.path.join('./data/img/up_rgb/'+img_list[i]+'.png'),img[inter_val[0]:inter_val[1],:,:])

        img = cv2.imread(infra_path+img_list[i]+'.png')
        img1 = np.resize(np.where(img[:,:,0]==0,mean_img[:,:,3],img[:,:,0]),(img.shape[0],img.shape[1],1))
        img2 = np.resize(np.where(img[:,:,1]==0,mean_img[:,:,4],img[:,:,1]),(img.shape[0],img.shape[1],1))
        img3 = np.resize(np.where(img[:,:,2]==0,mean_img[:,:,5],img[:,:,2]),(img.shape[0],img.shape[1],1))
        img = np.concatenate((img1,img2,img3), axis=2)
        cv2.imwrite(os.path.join('./data/img/up_infra/'+img_list[i]+'.png'),img[inter_val[0]:inter_val[1],:,:])
        #print(asd)

def update_npy(elemt_rmv,inter_val):
    '''calculate mean numpy array from a dir of numpy array of banklines'''
    inter_val = [10,2232]
    elemt_rmv = 2
    npy_path = os.path.join('./data/img/line_npy/')
    npy_list = [f for f in listdir(npy_path) if isfile(join(npy_path, f))]
    npy_list = [int(f.split('.')[0]) for f in npy_list]
    npy_list.sort()
    npy_list = [str(f) for f in npy_list]
    print('reading data from .......')
    print(npy_list)
    for i in range(len(npy_list)):
        line_npy = np.load(npy_path+npy_list[i]+'.npy')
        line_npy = line_npy[inter_val[0]:inter_val[1],:]
        np.save(os.path.join('./data/img/up_npy/'+npy_list[i]+'.npy'),line_npy)

#for i in range(elemt_rmv):
#   npy_list.pop()

""" line_npy = np.load(npy_path+npy_list[i]+'.npy')
#coor_mean = np.zeros((1,line_npy.shape[1]))
coor_mean = []
for i in range(len(npy_list)):
    line_npy = np.load(npy_path+npy_list[i]+'.npy')
    line_npy = line_npy[inter_val[0]:inter_val[1],:]
    np.save('./data/img/up_npy/'+npy_list[i]+'.npy',line_npy)
    coor_mean.append(line_npy)

arr = np.resize(np.asarray(coor_mean),(-1,2))
coor_mean = np.mean(arr,axis=0)
coor_std = np.std(arr,axis=0)
print('writing mean and std numpy arrays .......')
np.save('./data/mean_img/line_mean.npy',coor_mean)
np.save('./data/mean_img/line_std.npy',coor_std)
print(coor_mean,coor_std) """

def update_bin_mask(inter_val):

    msk_path = os.path.join('./data/img/png/')
    msk_list = [f for f in listdir(msk_path) if isfile(join(msk_path, f))]
    msk_list = [int(f.split('.')[0]) for f in msk_list]
    msk_list.sort()
    msk_list = [str(f) for f in msk_list]

    print('writing msk .....')
    for i in range(len(msk_list)):
        msk_img = cv2.imread(msk_path+msk_list[i]+'.png',0)
        cv2.imwrite(os.path.join('./data/img/up_msk/'+msk_list[i]+'.png'),msk_img[inter_val[0]:inter_val[1],:])

""" def comp_to_tfrec():
    

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        ###Returns a float_list from a float / double.
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
                
    def write_data():
        writer = tf.io.TFRecordWriter('./data/tfrecord/'+ 'comp_tf' +'.tfrecords')

        rgb_path = './data/img/up_rgb/'
        rgb_list = [f for f in listdir(rgb_path) if isfile(join(rgb_path, f))]
        rgb_list = [int(f.split('.')[0]) for f in rgb_list]
        rgb_list.sort()
        rgb_list = [str(f) for f in rgb_list]

        infra_path = './data/img/up_infra/'
        msk_path = './data/img/up_msk/'
        npy_path = './data/img/up_npy/'

        mean_coor = np.load('./data/mean_img/line_mean.npy')
        std_coor = np.load('./data/mean_img/line_std.npy')

        print('reading data from .....')
        print(rgb_list)

        temp_img = cv2.imread(rgb_path+rgb_list[0]+'.png')

        for i in range(temp_img.shape[0]):
            print(i)
            for j in range(len(rgb_list)):
                #print(rgb_list[j])
                rgb_img = cv2.imread(rgb_path+rgb_list[j]+'.png')
                rgb_img = rgb_img[i,:,:]
                rgb_img = rgb_img/255

                infra_img = cv2.imread(infra_path+rgb_list[j]+'.png')
                infra_img = infra_img[i,:,:]
                infra_img = infra_img/255

                msk_img = cv2.imread(msk_path+rgb_list[j]+'.png',0)
                msk_img = msk_img[i,:]
                msk_img = np.resize(msk_img,(745,1)) 
                msk_img = msk_img/255  

                input_img = np.concatenate((rgb_img,infra_img,msk_img),axis=1)

                line_npy = np.load(npy_path+rgb_list[j]+'.npy')
                line_npy = line_npy[i,:]

                if j == 0 :
                    prev_npy = line_npy
                    bin_npy = np.asarray([0,0])
                else :
                    bin_npy = np.where(prev_npy == line_npy, 0, 1)
                    prev_npy = line_npy
                
                line_npy = (line_npy-mean_coor)/std_coor

                #print(input_img.astype(float).type)
                input_img = np.asarray(input_img,dtype = np.float32)
                line_npy = np.asarray(line_npy,dtype =  np.float32)
                bin_npy = np.asarray(bin_npy,dtype =  np.float32)
                year_id = np.asarray(int(rgb_list[j]),dtype =  np.float32)

                feature = {
                            'input_tensor': _bytes_feature(input_img.tostring()),
                            'reg_coor': _bytes_feature(line_npy.tostring()),
                            'bin_label': _bytes_feature(bin_npy.tostring()),
                            'year_id' : _bytes_feature(year_id.tostring())
                            }

                example = tf.train.Example(features=tf.train.Features(feature=feature))

                writer.write(example.SerializeToString())

        writer.close()
        sys.stdout.flush()   

    write_data()  """
    
def for_cegis():
    with fiona.open('./data/cegis_19/CEGIS_2019_5.shp', "r") as shapefile:
        #with fiona.open('./data/img/shape_files/2019/201901.shp', "r") as shapefile:
        #print(type(shapefile))
        #print(asd)
        shapes = [feature["geometry"] for feature in shapefile]
        #print(shapes)

    temp_tif_path = './data/img/temp.tif'
    with rasterio.open(temp_tif_path) as src:
        temp_t = src.read()
        #print(temp_t)
        out_image, out_transform = rasterio.mask.mask(src, shapes)
        #print(out_image)
    
    img = wrt_bin_mask(out_image, 'cegis_19')

    left_lis, right_lis = mask_to_bnk_list(img)

    #save_infra = save_img_from_tif(tif_path, 'infra',True, file_id)

    tif_img = save_img_from_tif('./data/img/finaltif/201901.tif', 'rgb',True, 'cegis_19')

    for i in range(tif_img.shape[0]):
        if left_lis[i] != 0 and right_lis[i] != 0:
            tif_img[i,left_lis[i],0] = 255
            tif_img[i,left_lis[i],1] = 255
            tif_img[i,left_lis[i],2] = 255
            tif_img[i,right_lis[i],0] = 255
            tif_img[i,right_lis[i],1] = 255
            tif_img[i,right_lis[i],2] = 255

    print('''writing rgb with lines images ......''')
    cv2.imwrite('./data/img/shp_mask/'+'cegis_19'+'.png', tif_img)

def line_npy_in_imgs():
    '''given a directory of numpy lines, write images with white lines according to
    the numpy line coordinates'''
    npy_path = os.path.join('./data/img/up_npy/')
    rgb_path = os.path.join('./data/img/up_rgb/')
    rgb_list = [f for f in listdir(rgb_path) if isfile(join(rgb_path, f))]
    rgb_list = [int(f.split('.')[0]) for f in rgb_list]
    rgb_list.sort()
    rgb_list = [str(f) for f in rgb_list]
    #print(rgb_list)
    
    for i in range(len(rgb_list)):
        img = cv2.imread(rgb_path+rgb_list[i]+'.png')
        npy_line = np.load(npy_path+rgb_list[i]+'.npy')
        for j in range(img.shape[0]):
            for k in range(2):
                img[j,int(npy_line[j,k]),:] = [255,255,255]
        cv2.imwrite(os.path.join('./data/img/up_lines_imgs/'+rgb_list[i]+'.png'),img)


def write_lines(strt_year,val_split):

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    strt = strt_year
    val_split = val_split
    npy_path = os.path.join('./data/img/up_npy/')
    rgb_path = os.path.join('./data/img/up_rgb/')
    rgb_list = [f for f in listdir(rgb_path) if isfile(join(rgb_path, f))]
    rgb_list = [int(f.split('.')[0]) for f in rgb_list]
    rgb_list.sort()
    rgb_list = [str(f) for f in rgb_list]
    #print(rgb_list)
    #rgb_list = rgb_list[]

    temp_img = cv2.imread(rgb_path+rgb_list[0]+'.png')
    sum_writer = SummaryWriter()
    writer = tf.io.TFRecordWriter(os.path.join('./data/tfrecord/'+ 'lines_'+str(strt)+'_'+str(val_split)+'.tfrecords'))

    lft_full_diff = []
    rgt_full_diff = []

    lft_full_ins = []
    rgt_full_ins = []

    lft_full_mean = []
    rgt_full_mean = []
    lft_full_std = []
    rgt_full_std = []

    lft_diff_full_mean = []
    rgt_diff_full_mean = []
    lft_diff_full_std = []
    rgt_diff_full_std = []

    global_j_count = 0
    for i in range(temp_img.shape[0]):
        print(i)
        left_reach_diff_values = []
        right_reach_diff_values = []

        left_reach_inputs = []
        right_reach_inputs = []
        for j in range(len(rgb_list)):
            year_id = np.asarray(int(rgb_list[j]),dtype =  np.float32)
            reach_id = np.asarray(int(i),dtype =  np.float32)
            line_npy = np.load(npy_path+rgb_list[j]+'.npy')
            line_npy = line_npy[i,:]
            line_npy = np.asarray(line_npy,dtype =  np.float32)

            if j == 0 :
                prev_npy = line_npy
                bin_npy = np.asarray([0,0])
                lft_reach_diff = 0 
                rgt_reach_diff = 0
            else :
                lft_reach_diff = line_npy[0] - prev_npy[0]
                rgt_reach_diff = line_npy[1] - prev_npy[1]

                bin_npy[0] = np.where(prev_npy[0] > line_npy[0], 1, 0)
                bin_npy[1] = np.where(prev_npy[1] < line_npy[1], 1, 0)
                prev_npy = line_npy
            
            bin_npy = np.asarray(bin_npy,dtype =  np.float32)

            """ print(line_npy)
            print(year_id)
            print(reach_id)
            print(bin_npy) """

            sum_writer.add_scalar('Left_trend/reach_'+str(i),line_npy[0],j)
            sum_writer.add_scalar('Right_trend/reach_'+str(i),line_npy[1],j)

            #sum_writer.add_scalar('Left_bin/reach_'+str(i),bin_npy[0],j)
            #sum_writer.add_scalar('Right_bn/reach_'+str(i),bin_npy[1],j)

            sum_writer.add_scalar('Left_diff/reach_'+str(i),lft_reach_diff,j)
            sum_writer.add_scalar('Right_diff/reach_'+str(i),rgt_reach_diff,j)

            sum_writer.add_scalars('complete_trend',{'left':line_npy[0],'right':line_npy[1]},global_j_count)

            if j != 0 :
                left_reach_diff_values.append(lft_reach_diff)
                right_reach_diff_values.append(rgt_reach_diff)

            reach_diff = [lft_reach_diff, rgt_reach_diff]
            reach_diff = np.asarray(reach_diff, dtype=np.float32)
            
            left_reach_inputs.append(line_npy[0])
            right_reach_inputs.append(line_npy[1])

            feature = {
                        'reg_coor': _bytes_feature(line_npy.tostring()),
                        'year_id' : _bytes_feature(year_id.tostring()),
                        'reach_id': _bytes_feature(reach_id.tostring()),
                        'bin_class': _bytes_feature(bin_npy.tostring()),
                        'reach_diff':_bytes_feature(reach_diff.tostring())
                        }

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())
            global_j_count += 1

        lft_full_diff.extend(left_reach_diff_values)
        rgt_full_diff.extend(right_reach_diff_values)

        left_reach_diff_values = np.asarray(left_reach_diff_values)
        right_reach_diff_values = np.asarray(right_reach_diff_values)

        sum_writer.add_histogram('Left_diff/reach_'+str(i),left_reach_diff_values,i,bins='auto')
        sum_writer.add_histogram('Right_diff/reach_'+str(i),right_reach_diff_values,i,bins='auto')
        sum_writer.add_histogram('Left_Diff_across_reach',left_reach_diff_values,i,bins='auto')
        sum_writer.add_histogram('Right_Diff_hist_across_reach',right_reach_diff_values,i,bins='auto')


        lft_rch_diff_train = left_reach_diff_values[:-(val_split)]
        rgt_rch_diff_train = right_reach_diff_values[:-(val_split)]
        lft_rch_diff_train = lft_rch_diff_train[strt:]
        rgt_rch_diff_train = rgt_rch_diff_train[strt:]
        lft_rch_diff_val = left_reach_diff_values[-(val_split):]
        rgt_rch_diff_val = right_reach_diff_values[-(val_split):]

        lft_rch_diff_mean = np.mean(lft_rch_diff_train,axis=0)
        rgt_rch_diff_mean = np.mean(rgt_rch_diff_train,axis=0)
        lft_rch_diff_std = np.std(lft_rch_diff_train,axis=0)
        rgt_rch_diff_std = np.std(rgt_rch_diff_train,axis=0)

        lft_diff_full_mean.append(lft_rch_diff_mean)
        rgt_diff_full_mean.append(rgt_rch_diff_mean)
        lft_diff_full_std.append(lft_rch_diff_std)
        rgt_diff_full_std.append(rgt_rch_diff_std)
        
        standd_lft_rch_diff = (lft_rch_diff_train - lft_rch_diff_mean)/lft_rch_diff_std
        standd_rgt_rch_diff = (rgt_rch_diff_train - rgt_rch_diff_mean)/rgt_rch_diff_std
        lft_rch_diff_sddval = (lft_rch_diff_val - lft_rch_diff_mean)/lft_rch_diff_std
        rgt_rch_diff_sddval = (rgt_rch_diff_val - rgt_rch_diff_mean)/rgt_rch_diff_std


        for j in range(standd_lft_rch_diff.shape[0]):
            sum_writer.add_scalar('Left_train_diff_unscld/reach_'+str(i),lft_rch_diff_train[j],j)
            sum_writer.add_scalar('Right_train_diff_unscld/reach_'+str(i),rgt_rch_diff_train[j],j)
            sum_writer.add_scalar('Left_train_diff_sdd/reach_'+str(i),standd_lft_rch_diff[j],j)
            sum_writer.add_scalar('Right_train_diff_sdd/reach_'+str(i),standd_rgt_rch_diff[j],j)

        for j in range(lft_rch_diff_sddval.shape[0]):
            sum_writer.add_scalar('Left_val_diff_unscld/reach_'+str(i),lft_rch_diff_val[j],j)
            sum_writer.add_scalar('Right_val_diff_unscld/reach_'+str(i),rgt_rch_diff_val[j],j)
            sum_writer.add_scalar('Left_val_diff_sdd/reach_'+str(i),lft_rch_diff_sddval[j],j)
            sum_writer.add_scalar('Right_val_diff_sdd/reach_'+str(i),rgt_rch_diff_sddval[j],j)



        sum_writer.add_histogram('Left_Diff_Standardized/reach_'+str(i),standd_lft_rch_diff,i,bins='auto')
        sum_writer.add_histogram('Right_Diff_Standardized/reach_'+str(i),standd_rgt_rch_diff,i,bins='auto')
        sum_writer.add_histogram('Left_Diff_Standardized_across_reach',standd_lft_rch_diff,i,bins='auto')
        sum_writer.add_histogram('Right_Diff_Standardized_across_reach',standd_rgt_rch_diff,i,bins='auto')
        
        sum_writer.add_histogram('Left_val_Diff_Standardized_across_reach',lft_rch_diff_sddval,i,bins='auto')
        sum_writer.add_histogram('Right_val_Diff_Standardized_across_reach',rgt_rch_diff_sddval,i,bins='auto')
 


        if i == 0 :
            stdd_trian_lft_diff = standd_lft_rch_diff
            stdd_trin_rgt_diff = standd_rgt_rch_diff
        else :
            stdd_trian_lft_diff = np.concatenate((stdd_trian_lft_diff,standd_lft_rch_diff))
            stdd_trin_rgt_diff = np.concatenate((stdd_trin_rgt_diff,standd_rgt_rch_diff)) 


        lft_full_ins.extend(left_reach_inputs)
        rgt_full_ins.extend(right_reach_inputs)
        
        left_reach_inputs = np.asarray(left_reach_inputs)
        right_reach_inputs = np.asarray(right_reach_inputs)

        sum_writer.add_histogram('Left_input/reach_'+str(i),left_reach_inputs,i,bins='auto')
        sum_writer.add_histogram('Right_input/reach_'+str(i),right_reach_inputs,i,bins='auto')
        sum_writer.add_histogram('Left_inp_across_reach',left_reach_inputs,i,bins='auto')
        sum_writer.add_histogram('Right_inp_hist_across_reach',right_reach_inputs,i,bins='auto')


        lft_rch_inp_train = left_reach_inputs[:-(val_split+1)]
        rgt_rch_inp_train = right_reach_inputs[:-(val_split+1)]
        lft_rch_inp_train = lft_rch_inp_train[strt:]
        rgt_rch_inp_train = rgt_rch_inp_train[strt:]
        lft_rch_inp_val = left_reach_inputs[-(val_split+1):]
        rgt_rch_inp_val = left_reach_inputs[-(val_split+1):]
        
        lft_reach_inp_mean = np.mean(lft_rch_inp_train,axis=0)
        rgt_reach_inp_mean = np.mean(rgt_rch_inp_train,axis=0)

        lft_reach_inp_std = np.std(lft_rch_inp_train,axis=0)
        rgt_reach_inp_std = np.std(rgt_rch_inp_train,axis=0)

        lft_full_mean.append(lft_reach_inp_mean)
        rgt_full_mean.append(rgt_reach_inp_mean)
        lft_full_std.append(lft_reach_inp_std)
        rgt_full_std.append(rgt_reach_inp_std)

        standd_lft_rch_inp = (lft_rch_inp_train - lft_reach_inp_mean)/lft_reach_inp_std
        standd_rgt_rch_inp = (rgt_rch_inp_train - rgt_reach_inp_mean)/rgt_reach_inp_std
        standd_lft_rch_inp_val = (lft_rch_inp_val - lft_reach_inp_mean)/lft_reach_inp_std
        standd_rgt_rch_inp_val = (rgt_rch_inp_val - rgt_reach_inp_mean)/rgt_reach_inp_std

        for j in range(standd_lft_rch_inp.shape[0]):
            sum_writer.add_scalar('Left_train_inp_unscld/reach_'+str(i),lft_rch_inp_train[j],j)
            sum_writer.add_scalar('Right_train_inp_unscld/reach_'+str(i),rgt_rch_inp_train[j],j)
            sum_writer.add_scalar('Left_train_inp_sdd/reach_'+str(i),standd_lft_rch_inp[j],j)
            sum_writer.add_scalar('Right_train_inp_sdd/reach_'+str(i),standd_rgt_rch_inp[j],j)

        for j in range(lft_rch_diff_sddval.shape[0]):
            sum_writer.add_scalar('Left_val_inp_unscld/reach_'+str(i),lft_rch_inp_val[j],j)
            sum_writer.add_scalar('Right_val_inp_unscld/reach_'+str(i),rgt_rch_inp_val[j],j)
            sum_writer.add_scalar('Left_val_inp_sdd/reach_'+str(i),standd_lft_rch_inp_val[j],j)
            sum_writer.add_scalar('Right_val_inp_sdd/reach_'+str(i),standd_rgt_rch_inp_val[j],j)




        sum_writer.add_histogram('Left_input_Standardized/reach_'+str(i),standd_lft_rch_inp,i,bins='auto')
        sum_writer.add_histogram('Right_input_Standardized/reach_'+str(i),standd_rgt_rch_inp,i,bins='auto')
        sum_writer.add_histogram('Left_input_Standardized_across_reach',standd_lft_rch_inp,i,bins='auto')
        sum_writer.add_histogram('Right_input_Standardized_across_reach',standd_rgt_rch_inp,i,bins='auto')
        sum_writer.add_histogram('Left_input_val_Standardized_across_reach',standd_lft_rch_inp_val,i,bins='auto')
        sum_writer.add_histogram('Right_input_val_Standardized_across_reach',standd_rgt_rch_inp_val,i,bins='auto')

        if i == 0 :
            stdd_trian_lft_inp = standd_lft_rch_inp
            stdd_trin_rgt_inp = standd_rgt_rch_inp
        else :
            stdd_trian_lft_inp = np.concatenate((stdd_trian_lft_inp,standd_lft_rch_inp))
            stdd_trin_rgt_inp = np.concatenate((stdd_trin_rgt_inp,standd_rgt_rch_inp)) 

        #print(asd)
    lft_full_mean = np.asarray(lft_full_mean)
    rgt_full_mean = np.asarray(rgt_full_mean)
    lft_full_std = np.asarray(lft_full_std) 
    rgt_full_std = np.asarray(rgt_full_std)

    stdd_npy_path = os.path.join('./data/trns_npys/')
    ###save left and right both mean and std 
    np.save(stdd_npy_path+'lft_inp_mean_'+str(strt)+'_'+str(val_split)+'.npy',lft_full_mean)
    np.save(stdd_npy_path+'rgt_inp_mean_'+str(strt)+'_'+str(val_split)+'.npy',rgt_full_mean)
    np.save(stdd_npy_path+'lft_inp_std_'+str(strt)+'_'+str(val_split)+'.npy',lft_full_std)
    np.save(stdd_npy_path+'rgt_inp_std_'+str(strt)+'_'+str(val_split)+'.npy',rgt_full_std)


    
    lft_diff_full_mean = np.asarray(lft_diff_full_mean)
    rgt_diff_full_mean = np.asarray(rgt_diff_full_mean)
    lft_diff_full_std = np.asarray(lft_diff_full_std) 
    rgt_diff_full_std = np.asarray(rgt_diff_full_std)

    ###save left and right both mean and std 
    np.save(stdd_npy_path+'lft_out_mean_'+str(strt)+'_'+str(val_split)+'.npy',lft_diff_full_mean)
    np.save(stdd_npy_path+'rgt_out_mean_'+str(strt)+'_'+str(val_split)+'.npy',rgt_diff_full_mean)
    np.save(stdd_npy_path+'lft_out_std_'+str(strt)+'_'+str(val_split)+'.npy',lft_diff_full_std)
    np.save(stdd_npy_path+'rgt_out_std_'+str(strt)+'_'+str(val_split)+'.npy',rgt_diff_full_std)




    lft_full_ins = np.asarray(lft_full_ins)
    rgt_full_ins = np.asarray(rgt_full_ins)
    sum_writer.add_histogram('Unscaled_inputs/left_full_reach',lft_full_ins,0,bins='auto')
    sum_writer.add_histogram('Unscaled_inputs/right_full_reach',rgt_full_ins,0,bins='auto')
    dataset_inputs = np.concatenate((lft_full_ins,rgt_full_ins))
    sum_writer.add_histogram('Unscaled_inputs/full_reach_both_banks',dataset_inputs,0,bins='auto')

    sum_writer.add_histogram('standardized_train_inputs/left_full_reach',stdd_trian_lft_inp,0,bins='auto')
    sum_writer.add_histogram('standardized_train_inputs/right_full_reach',stdd_trin_rgt_inp,0,bins='auto')
    stdd_train_dataset_inps = np.concatenate((stdd_trian_lft_inp,stdd_trin_rgt_inp))
    sum_writer.add_histogram('standardized_train_inputs/full_reach_both_banks',stdd_train_dataset_inps,0,bins='auto')

    lft_full_diff = np.asarray(lft_full_diff)
    rgt_full_diff = np.asarray(rgt_full_diff)
    sum_writer.add_histogram('Difference_outputs/left_full',lft_full_diff,0,bins='auto')
    sum_writer.add_histogram('Difference_outputs/right_full',rgt_full_diff,0,bins='auto')
    dataset_diff = np.concatenate((lft_full_diff,rgt_full_diff))
    sum_writer.add_histogram('Difference_outputs/full_reach_both_banks_',dataset_diff,0,bins='auto')
    
    sum_writer.add_histogram('standardized_Diff_outputs/left_full_reach',stdd_trian_lft_diff,0,bins='auto')
    sum_writer.add_histogram('standardized_Diff_outputs/right_full_reach',stdd_trin_rgt_diff,0,bins='auto')
    stdd_train_dataset_outputs = np.concatenate((stdd_trian_lft_diff,stdd_trin_rgt_diff))
    sum_writer.add_histogram('standardized_Diff_outputs/full_reach_both_banks',stdd_train_dataset_outputs,0,bins='auto')


    #print(asd)
    writer.close()
    sys.stdout.flush() 


def write_lines(strt_year,val_split,time_step,reach_strt,reach_end):

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    strt = strt_year
    val_split = val_split
    reach_nums = reach_end - reach_strt


    npy_path = os.path.join('./data/img/up_npy/')
    rgb_path = os.path.join('./data/img/up_rgb/')
    rgb_list = [f for f in listdir(rgb_path) if isfile(join(rgb_path, f))]
    rgb_list = [int(f.split('.')[0]) for f in rgb_list]
    rgb_list.sort()
    rgb_list = [str(f) for f in rgb_list]
    #print(rgb_list)
    #rgb_list = rgb_list[]

    temp_img = cv2.imread(rgb_path+rgb_list[0]+'.png')
    sum_writer = SummaryWriter()
    writer = tf.io.TFRecordWriter(os.path.join('./data/tfrecord/'+ 'lines_'+str(strt)+'_'+str(val_split)+'.tfrecords'))

    lft_full_diff = []
    rgt_full_diff = []

    lft_full_ins = []
    rgt_full_ins = []

    lft_full_mean = []
    rgt_full_mean = []
    lft_full_std = []
    rgt_full_std = []

    lft_diff_full_mean = []
    rgt_diff_full_mean = []
    lft_diff_full_std = []
    rgt_diff_full_std = []

    global_j_count = 0
    for i in range(reach_nums):
        print(i)
        left_reach_diff_values = []
        right_reach_diff_values = []

        left_reach_inputs = []
        right_reach_inputs = []
        for j in range(len(rgb_list)):
            year_id = np.asarray(int(rgb_list[j]),dtype =  np.float32)
            reach_id = np.asarray(int(i),dtype =  np.float32)
            line_npy = np.load(npy_path+rgb_list[j]+'.npy')
            line_npy = line_npy[i,:]
            line_npy = np.asarray(line_npy,dtype =  np.float32)

            if j == 0 :
                prev_npy = line_npy
                bin_npy = np.asarray([0,0])
                lft_reach_diff = 0 
                rgt_reach_diff = 0
            else :
                lft_reach_diff = line_npy[0] - prev_npy[0]
                rgt_reach_diff = line_npy[1] - prev_npy[1]

                bin_npy[0] = np.where(prev_npy[0] > line_npy[0], 1, 0)
                bin_npy[1] = np.where(prev_npy[1] < line_npy[1], 1, 0)
                prev_npy = line_npy
            
            bin_npy = np.asarray(bin_npy,dtype =  np.float32)

            """ print(line_npy)
            print(year_id)
            print(reach_id)
            print(bin_npy) """

            sum_writer.add_scalar('Left_trend/reach_'+str(i),line_npy[0],j)
            sum_writer.add_scalar('Right_trend/reach_'+str(i),line_npy[1],j)

            #sum_writer.add_scalar('Left_bin/reach_'+str(i),bin_npy[0],j)
            #sum_writer.add_scalar('Right_bn/reach_'+str(i),bin_npy[1],j)

            sum_writer.add_scalar('Left_diff/reach_'+str(i),lft_reach_diff,j)
            sum_writer.add_scalar('Right_diff/reach_'+str(i),rgt_reach_diff,j)

            sum_writer.add_scalars('complete_trend',{'left':line_npy[0],'right':line_npy[1]},global_j_count)

            if j != 0 :
                left_reach_diff_values.append(lft_reach_diff)
                right_reach_diff_values.append(rgt_reach_diff)

            reach_diff = [lft_reach_diff, rgt_reach_diff]
            reach_diff = np.asarray(reach_diff, dtype=np.float32)
            
            left_reach_inputs.append(line_npy[0])
            right_reach_inputs.append(line_npy[1])

            feature = {
                        'reg_coor': _bytes_feature(line_npy.tostring()),
                        'year_id' : _bytes_feature(year_id.tostring()),
                        'reach_id': _bytes_feature(reach_id.tostring()),
                        'bin_class': _bytes_feature(bin_npy.tostring()),
                        'reach_diff':_bytes_feature(reach_diff.tostring())
                        }

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())
            global_j_count += 1

        lft_full_diff.extend(left_reach_diff_values)
        rgt_full_diff.extend(right_reach_diff_values)

        left_reach_diff_values = np.asarray(left_reach_diff_values)
        right_reach_diff_values = np.asarray(right_reach_diff_values)

        sum_writer.add_histogram('Left_diff/reach_'+str(i),left_reach_diff_values,i,bins='auto')
        sum_writer.add_histogram('Right_diff/reach_'+str(i),right_reach_diff_values,i,bins='auto')
        sum_writer.add_histogram('Left_Diff_across_reach',left_reach_diff_values,i,bins='auto')
        sum_writer.add_histogram('Right_Diff_hist_across_reach',right_reach_diff_values,i,bins='auto')


        lft_rch_diff_train = left_reach_diff_values[:-(val_split)]
        rgt_rch_diff_train = right_reach_diff_values[:-(val_split)]
        lft_rch_diff_train = lft_rch_diff_train[strt:]
        rgt_rch_diff_train = rgt_rch_diff_train[strt:]
        lft_rch_diff_val = left_reach_diff_values[-(val_split):]
        rgt_rch_diff_val = right_reach_diff_values[-(val_split):]

        #lft_rch_diff_mean = np.mean(lft_rch_diff_train,axis=0)
        #rgt_rch_diff_mean = np.mean(rgt_rch_diff_train,axis=0)
        #lft_rch_diff_std = np.std(lft_rch_diff_train,axis=0)
        #rgt_rch_diff_std = np.std(rgt_rch_diff_train,axis=0)

        """ lft_diff_full_mean.append(lft_rch_diff_mean)
        rgt_diff_full_mean.append(rgt_rch_diff_mean)
        lft_diff_full_std.append(lft_rch_diff_std)
        rgt_diff_full_std.append(rgt_rch_diff_std) """
        
        standd_lft_rch_diff = (lft_rch_diff_train - lft_rch_diff_mean)/lft_rch_diff_std
        standd_rgt_rch_diff = (rgt_rch_diff_train - rgt_rch_diff_mean)/rgt_rch_diff_std
        lft_rch_diff_sddval = (lft_rch_diff_val - lft_rch_diff_mean)/lft_rch_diff_std
        rgt_rch_diff_sddval = (rgt_rch_diff_val - rgt_rch_diff_mean)/rgt_rch_diff_std


        for j in range(standd_lft_rch_diff.shape[0]):
            sum_writer.add_scalar('Left_train_diff_unscld/reach_'+str(i),lft_rch_diff_train[j],j)
            sum_writer.add_scalar('Right_train_diff_unscld/reach_'+str(i),rgt_rch_diff_train[j],j)
            sum_writer.add_scalar('Left_train_diff_sdd/reach_'+str(i),standd_lft_rch_diff[j],j)
            sum_writer.add_scalar('Right_train_diff_sdd/reach_'+str(i),standd_rgt_rch_diff[j],j)

        for j in range(lft_rch_diff_sddval.shape[0]):
            sum_writer.add_scalar('Left_val_diff_unscld/reach_'+str(i),lft_rch_diff_val[j],j)
            sum_writer.add_scalar('Right_val_diff_unscld/reach_'+str(i),rgt_rch_diff_val[j],j)
            sum_writer.add_scalar('Left_val_diff_sdd/reach_'+str(i),lft_rch_diff_sddval[j],j)
            sum_writer.add_scalar('Right_val_diff_sdd/reach_'+str(i),rgt_rch_diff_sddval[j],j)



        sum_writer.add_histogram('Left_Diff_Standardized/reach_'+str(i),standd_lft_rch_diff,i,bins='auto')
        sum_writer.add_histogram('Right_Diff_Standardized/reach_'+str(i),standd_rgt_rch_diff,i,bins='auto')
        sum_writer.add_histogram('Left_Diff_Standardized_across_reach',standd_lft_rch_diff,i,bins='auto')
        sum_writer.add_histogram('Right_Diff_Standardized_across_reach',standd_rgt_rch_diff,i,bins='auto')
        
        sum_writer.add_histogram('Left_val_Diff_Standardized_across_reach',lft_rch_diff_sddval,i,bins='auto')
        sum_writer.add_histogram('Right_val_Diff_Standardized_across_reach',rgt_rch_diff_sddval,i,bins='auto')
 


        if i == 0 :
            stdd_trian_lft_diff = standd_lft_rch_diff
            stdd_trin_rgt_diff = standd_rgt_rch_diff
        else :
            stdd_trian_lft_diff = np.concatenate((stdd_trian_lft_diff,standd_lft_rch_diff))
            stdd_trin_rgt_diff = np.concatenate((stdd_trin_rgt_diff,standd_rgt_rch_diff)) 


        lft_full_ins.extend(left_reach_inputs)
        rgt_full_ins.extend(right_reach_inputs)
        
        left_reach_inputs = np.asarray(left_reach_inputs)
        right_reach_inputs = np.asarray(right_reach_inputs)

        sum_writer.add_histogram('Left_input/reach_'+str(i),left_reach_inputs,i,bins='auto')
        sum_writer.add_histogram('Right_input/reach_'+str(i),right_reach_inputs,i,bins='auto')
        sum_writer.add_histogram('Left_inp_across_reach',left_reach_inputs,i,bins='auto')
        sum_writer.add_histogram('Right_inp_hist_across_reach',right_reach_inputs,i,bins='auto')


        lft_rch_inp_train = left_reach_inputs[:-(val_split+1)]
        rgt_rch_inp_train = right_reach_inputs[:-(val_split+1)]
        lft_rch_inp_train = lft_rch_inp_train[strt:]
        rgt_rch_inp_train = rgt_rch_inp_train[strt:]
        lft_rch_inp_val = left_reach_inputs[-(val_split+1):]
        rgt_rch_inp_val = left_reach_inputs[-(val_split+1):]
        
        lft_reach_inp_mean = np.mean(lft_rch_inp_train,axis=0)
        rgt_reach_inp_mean = np.mean(rgt_rch_inp_train,axis=0)

        lft_reach_inp_std = np.std(lft_rch_inp_train,axis=0)
        rgt_reach_inp_std = np.std(rgt_rch_inp_train,axis=0)

        lft_full_mean.append(lft_reach_inp_mean)
        rgt_full_mean.append(rgt_reach_inp_mean)
        lft_full_std.append(lft_reach_inp_std)
        rgt_full_std.append(rgt_reach_inp_std)

        standd_lft_rch_inp = (lft_rch_inp_train - lft_reach_inp_mean)/lft_reach_inp_std
        standd_rgt_rch_inp = (rgt_rch_inp_train - rgt_reach_inp_mean)/rgt_reach_inp_std
        standd_lft_rch_inp_val = (lft_rch_inp_val - lft_reach_inp_mean)/lft_reach_inp_std
        standd_rgt_rch_inp_val = (rgt_rch_inp_val - rgt_reach_inp_mean)/rgt_reach_inp_std

        for j in range(standd_lft_rch_inp.shape[0]):
            sum_writer.add_scalar('Left_train_inp_unscld/reach_'+str(i),lft_rch_inp_train[j],j)
            sum_writer.add_scalar('Right_train_inp_unscld/reach_'+str(i),rgt_rch_inp_train[j],j)
            sum_writer.add_scalar('Left_train_inp_sdd/reach_'+str(i),standd_lft_rch_inp[j],j)
            sum_writer.add_scalar('Right_train_inp_sdd/reach_'+str(i),standd_rgt_rch_inp[j],j)

        for j in range(lft_rch_diff_sddval.shape[0]):
            sum_writer.add_scalar('Left_val_inp_unscld/reach_'+str(i),lft_rch_inp_val[j],j)
            sum_writer.add_scalar('Right_val_inp_unscld/reach_'+str(i),rgt_rch_inp_val[j],j)
            sum_writer.add_scalar('Left_val_inp_sdd/reach_'+str(i),standd_lft_rch_inp_val[j],j)
            sum_writer.add_scalar('Right_val_inp_sdd/reach_'+str(i),standd_rgt_rch_inp_val[j],j)




        sum_writer.add_histogram('Left_input_Standardized/reach_'+str(i),standd_lft_rch_inp,i,bins='auto')
        sum_writer.add_histogram('Right_input_Standardized/reach_'+str(i),standd_rgt_rch_inp,i,bins='auto')
        sum_writer.add_histogram('Left_input_Standardized_across_reach',standd_lft_rch_inp,i,bins='auto')
        sum_writer.add_histogram('Right_input_Standardized_across_reach',standd_rgt_rch_inp,i,bins='auto')
        sum_writer.add_histogram('Left_input_val_Standardized_across_reach',standd_lft_rch_inp_val,i,bins='auto')
        sum_writer.add_histogram('Right_input_val_Standardized_across_reach',standd_rgt_rch_inp_val,i,bins='auto')

        if i == 0 :
            stdd_trian_lft_inp = standd_lft_rch_inp
            stdd_trin_rgt_inp = standd_rgt_rch_inp
        else :
            stdd_trian_lft_inp = np.concatenate((stdd_trian_lft_inp,standd_lft_rch_inp))
            stdd_trin_rgt_inp = np.concatenate((stdd_trin_rgt_inp,standd_rgt_rch_inp)) 

        #print(asd)
    lft_full_mean = np.asarray(lft_full_mean)
    rgt_full_mean = np.asarray(rgt_full_mean)
    lft_full_std = np.asarray(lft_full_std) 
    rgt_full_std = np.asarray(rgt_full_std)

    stdd_npy_path = os.path.join('./data/trns_npys/')
    ###save left and right both mean and std 
    np.save(stdd_npy_path+'lft_inp_mean_'+str(strt)+'_'+str(val_split)+'.npy',lft_full_mean)
    np.save(stdd_npy_path+'rgt_inp_mean_'+str(strt)+'_'+str(val_split)+'.npy',rgt_full_mean)
    np.save(stdd_npy_path+'lft_inp_std_'+str(strt)+'_'+str(val_split)+'.npy',lft_full_std)
    np.save(stdd_npy_path+'rgt_inp_std_'+str(strt)+'_'+str(val_split)+'.npy',rgt_full_std)


    
    lft_diff_full_mean = np.asarray(lft_diff_full_mean)
    rgt_diff_full_mean = np.asarray(rgt_diff_full_mean)
    lft_diff_full_std = np.asarray(lft_diff_full_std) 
    rgt_diff_full_std = np.asarray(rgt_diff_full_std)

    ###save left and right both mean and std 
    np.save(stdd_npy_path+'lft_out_mean_'+str(strt)+'_'+str(val_split)+'.npy',lft_diff_full_mean)
    np.save(stdd_npy_path+'rgt_out_mean_'+str(strt)+'_'+str(val_split)+'.npy',rgt_diff_full_mean)
    np.save(stdd_npy_path+'lft_out_std_'+str(strt)+'_'+str(val_split)+'.npy',lft_diff_full_std)
    np.save(stdd_npy_path+'rgt_out_std_'+str(strt)+'_'+str(val_split)+'.npy',rgt_diff_full_std)




    lft_full_ins = np.asarray(lft_full_ins)
    rgt_full_ins = np.asarray(rgt_full_ins)
    sum_writer.add_histogram('Unscaled_inputs/left_full_reach',lft_full_ins,0,bins='auto')
    sum_writer.add_histogram('Unscaled_inputs/right_full_reach',rgt_full_ins,0,bins='auto')
    dataset_inputs = np.concatenate((lft_full_ins,rgt_full_ins))
    sum_writer.add_histogram('Unscaled_inputs/full_reach_both_banks',dataset_inputs,0,bins='auto')

    sum_writer.add_histogram('standardized_train_inputs/left_full_reach',stdd_trian_lft_inp,0,bins='auto')
    sum_writer.add_histogram('standardized_train_inputs/right_full_reach',stdd_trin_rgt_inp,0,bins='auto')
    stdd_train_dataset_inps = np.concatenate((stdd_trian_lft_inp,stdd_trin_rgt_inp))
    sum_writer.add_histogram('standardized_train_inputs/full_reach_both_banks',stdd_train_dataset_inps,0,bins='auto')

    lft_full_diff = np.asarray(lft_full_diff)
    rgt_full_diff = np.asarray(rgt_full_diff)
    sum_writer.add_histogram('Difference_outputs/left_full',lft_full_diff,0,bins='auto')
    sum_writer.add_histogram('Difference_outputs/right_full',rgt_full_diff,0,bins='auto')
    dataset_diff = np.concatenate((lft_full_diff,rgt_full_diff))
    sum_writer.add_histogram('Difference_outputs/full_reach_both_banks_',dataset_diff,0,bins='auto')
    
    sum_writer.add_histogram('standardized_Diff_outputs/left_full_reach',stdd_trian_lft_diff,0,bins='auto')
    sum_writer.add_histogram('standardized_Diff_outputs/right_full_reach',stdd_trin_rgt_diff,0,bins='auto')
    stdd_train_dataset_outputs = np.concatenate((stdd_trian_lft_diff,stdd_trin_rgt_diff))
    sum_writer.add_histogram('standardized_Diff_outputs/full_reach_both_banks',stdd_train_dataset_outputs,0,bins='auto')


    #print(asd)
    writer.close()
    sys.stdout.flush() 



def write_stdd_lines(strt_year,val_split):
    strt = strt_year


    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    val_split = val_split
    npy_path = os.path.join('./data/img/up_npy/')
    rgb_path = os.path.join('./data/img/up_rgb/')
    rgb_list = [f for f in listdir(rgb_path) if isfile(join(rgb_path, f))]
    rgb_list = [int(f.split('.')[0]) for f in rgb_list]
    rgb_list.sort()
    rgb_list = [str(f) for f in rgb_list]

    temp_img = cv2.imread(rgb_path+rgb_list[0]+'.png')

    writer = tf.io.TFRecordWriter(os.path.join('./data/tfrecord/'+ 'lines_sdd_'+str(strt)+'_'+str(val_split)+'.tfrecords'))
    stdd_npy_path = os.path.join('./data/trns_npys/')

    lft_inp_mean = np.load(stdd_npy_path+'lft_inp_mean_'+str(strt)+'_'+str(val_split)+'.npy')
    rgt_inp_mean = np.load(stdd_npy_path+'rgt_inp_mean_'+str(strt)+'_'+str(val_split)+'.npy')
    lft_inp_std = np.load(stdd_npy_path+'lft_inp_std_'+str(strt)+'_'+str(val_split)+'.npy')
    rgt_inp_std = np.load(stdd_npy_path+'rgt_inp_std_'+str(strt)+'_'+str(val_split)+'.npy')

    lft_diff_full_mean = np.load(stdd_npy_path+'lft_out_mean_'+str(strt)+'_'+str(val_split)+'.npy')
    rgt_diff_full_mean = np.load(stdd_npy_path+'rgt_out_mean_'+str(strt)+'_'+str(val_split)+'.npy')
    lft_diff_full_std = np.load(stdd_npy_path+'lft_out_std_'+str(strt)+'_'+str(val_split)+'.npy')
    rgt_diff_full_std = np.load(stdd_npy_path+'rgt_out_std_'+str(strt)+'_'+str(val_split)+'.npy')
    
    print(lft_inp_mean.shape)
    print(lft_diff_full_mean.shape)
    print(temp_img.shape[0])

    for i in range(temp_img.shape[0]):
        print(i)
        for j in range(len(rgb_list)):
            year_id = np.asarray(int(rgb_list[j]),dtype =  np.float32)
            reach_id = np.asarray(int(i),dtype =  np.float32)
            line_npy = np.load(npy_path+rgb_list[j]+'.npy')
            line_npy = line_npy[i,:]

            if j == 0 :
                prev_npy = line_npy
                bin_npy = np.asarray([0,0])
                lft_reach_diff = 0 
                rgt_reach_diff = 0
            else :
                lft_reach_diff = line_npy[0] - prev_npy[0]
                rgt_reach_diff = line_npy[1] - prev_npy[1]

                bin_npy[0] = np.where(prev_npy[0] > line_npy[0], 1, 0)
                bin_npy[1] = np.where(prev_npy[1] < line_npy[1], 1, 0)
                prev_npy = line_npy
            
            bin_npy = np.asarray(bin_npy,dtype =  np.float32)
            reach_diff = [lft_reach_diff, rgt_reach_diff]
            reach_diff = np.asarray(reach_diff, dtype=np.float32)

            lft_sdd_line = (line_npy[0] - lft_inp_mean[i]) / lft_inp_std[i]
            rgt_sdd_line = (line_npy[1] - rgt_inp_mean[i]) / rgt_inp_std[i]

            lft_sdd_diff = (reach_diff[0] - lft_diff_full_mean[i]) / lft_diff_full_std[i]
            rgt_sdd_diff = (reach_diff[1] - rgt_diff_full_mean[i]) / rgt_diff_full_std[i]

            line_npy = np.asarray([lft_sdd_line, rgt_sdd_line],dtype =  np.float32)
            reach_diff = np.asarray([lft_sdd_diff, rgt_sdd_diff],dtype =  np.float32)
            #print(line_npy.shape)
            #print(asd)

            feature = {
                        'sdd_input': _bytes_feature(line_npy.tostring()),
                        'year_id' : _bytes_feature(year_id.tostring()),
                        'reach_id': _bytes_feature(reach_id.tostring()),
                        'bin_class': _bytes_feature(bin_npy.tostring()),
                        'sdd_output':_bytes_feature(reach_diff.tostring())
                        }

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush() 

if __name__ == "__main__" :
    
    """ img = cv2.imread(os.path.join('.\\data\\img\\up_lines_imgs\\201901.png'))

    cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
    cv2.imshow('image1',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(asd) """
    #write_lines(strt_year=0,val_split=5)
    #write_stdd_lines(strt_year=0,val_split=5)
    #print(asd)
    """ npy_path = os.path.join('./data/img/up_npy/')
    rgb_path = os.path.join('./data/img/up_rgb/')
    rgb_list = [f for f in listdir(rgb_path) if isfile(join(rgb_path, f))]
    rgb_list = [int(f.split('.')[0]) for f in rgb_list]
    rgb_list.sort()
    rgb_list = [str(f) for f in rgb_list]

    temp_img = cv2.imread(rgb_path+'201901.png')


    full_list = []

    reach_len = range(890,975)

    for i in reach_len:
        print(i)
        row_list = []
        for j in range(len(rgb_list)):
            line_npy = np.load(npy_path+rgb_list[j]+'.npy')
            line_npy = line_npy[i,0]
            row_list.append(line_npy)

        full_list.append(row_list)
        temp_img[i,row_list[-2],:] = [255,255,255]

    full_list = np.asarray(full_list)
    print(full_list.shape)

    np.savetxt(os.path.join('./data/right_01.csv'), full_list, delimiter=",")
    cv2.imwrite(os.path.join('./data/right_01.png'),temp_img) """

    """
    for i in range(20,26,1):
        write_lines(strt_year=i,val_split=5)
        write_stdd_lines(strt_year=i,val_split=5) """
    #write_stdd_lines()
    #
    #line_npy_in_imgs()
    #save_img_from_tif(os.path.join('./data/img/final_rgb/198802.tif'),'infra',True,'198802')
    #for_cegis()
    #comp_to_tfrec()
    #update_bin_mask([10,2232])
    #update_npy(2,[10,2232])
    #up_rgb_infra([10,2232])
    
    #mean_tensor(2)
    #tif_path = os.path.join('./data/img/finaltif/')
    #batch_shp_to_data(tif_path)
    #wrt_temp_blank_tif()
    #shp_mask_tif("./data/img/shape_files/2019/201901.shp",'./data/img/finaltif/201901.tif')

    

    """ img = rasterio.open("./data/img/shp_mask/198801.tif")
    img = img.read()
    img = rasterio.plot.reshape_as_image(img)
    img = np.asarray(img, dtype = np.uint8)
    print(img.shape)
    cv2.imwrite('./data/img/shp_mask/198801.png',img[:,:,3]) """
    #apply_signal_denoising()
    #mean_tensor()
    """ mean_line = np.load('./data/mean_img/mean_line.npy')
    print(mean_line.shape)
    print(mean_line)
    line = np.tile(mean_line,(12,5,1,1))
    print(line.shape)
    print(line[0,0,:,:]) """
    
    """ data = rasterio.open('./data/198801.tif')
    img = data.read()
    img = rasterio.plot.reshape_as_image(img)
    img_np = np.asarray(img)
    #img = np.where(img_np!=0,255,0)
    img = np.uint8(img)

    cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
    cv2.imshow('image1',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() """

    
    #check_img_dist()
    #img.cv_write('./data/img/png/','new')
    #single_pix('./data/img/png/', './data/img/lines/')
    pass
