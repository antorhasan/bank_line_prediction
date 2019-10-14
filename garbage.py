import cv2
import numpy as np 




""" img = cv2.imread('./data/exp1/201601.png')

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
 """
def data_ag():
    """ path = path_sort('./data/exp1/198801.png')
    #path = path[0:2]
    data = []
    for i in range(len(path)):
        print(i) """
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