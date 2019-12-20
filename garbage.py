import cv2
import numpy as np 


""" left_lis_147 = [20110100,20120100,20130100,20140100,20150100,20160100,20170100,20180100,20190100]
right_lis_208 = [20110101,20120101,20130101,20140101,20150101,20160101,20170101,20180101,20190101]
right_lis = [201101,201201,201301,201401,201501,201601,201701,201801,201901]
right_lis = [201801,201901]
other = [19880161,19880171,19890171] """

#img = cv2.imread('./data/img/lines/198801.png')
img = cv2.imread('./data/img/result_imgs/label1.png')
#ctrl = 860
#img = img[7*256:8*256,ctrl-128:ctrl+128]
img = 255-img

#img = np.asarray(img)

#img = np.where(img==[0,255,255],[255,0,0],img)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i,j,0] == 255 and img[i,j,1] == 255 and img[i,j,2] == 0:
            img[i,j,0] = 0
            img[i,j,1] = 0
            img[i,j,2] = 255

#img = np.asarray(img, dtype=np.uint8)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('./data/img/result_imgs/color_17.png', img)
   
""" 
for i in range(len(right_lis)):

    #print(right_lis_208[i])
    img = cv2.imread('./data/img/lines/' + str(right_lis[i]) + '.png', 0)
    ctrl = 679
    img = img[0*256:1*256,ctrl-128:ctrl+128]
    img[0,147] = 255
    cv2.imwrite('./data/img/msk_corrected/' + str(right_lis[i]) +'01'+ '.png', img) """




""" 
def data_ag():
    path = path_sort('./data/exp1/198801.png')
    #path = path[0:2]
    data = []
    for i in range(len(path)):
        print(i)
    data = []
    img = cv2.imread('./data/exp1/198801.png',0)
    for j in range(img.shape[0]):
        for k in range(img.shape[1]):
            if img[j,k] == 255 :
                data.append(k)
    return data

data = data_ag()
print(data[0:50])
data = np.asarray(data)

data = np.where(data>505,data-505,505-data)

print(data[0:50])
 """


#write_data_f('normal_dis', 'train')
#write_data_f('normal_dis', 'val', 'val28')
#write_data_f('normal_dis', 'test', 'test28')


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