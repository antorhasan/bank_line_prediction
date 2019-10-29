import cv2
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_array


def imgs_mse(path_org, path_pred) :
    '''given two images where one is original bankline image in greyscale
    and the other is predicted line drawn over original image in red'''

    img_pred = cv2.imread(path_pred, 1)
    img_org = cv2.imread(path_org, 0)

    left_pred = []
    left_org = []

    right_pred = []
    right_org = []
    for i in range(img_org.shape[0]):
        coun = 0
        for j in range(img_org.shape[1]):
            if img_org[i,j] == [255] and coun == 0 :
                left_org.append(j)
                coun += 1
            elif img_org[i,j] == [255] and coun == 1 :
                right_org.append(j)

    for i in range(img_pred.shape[0]):
        coun = 0
        for j in range(img_pred.shape[1]):
            #print(img_pred[i,j].tolist())
            if img_pred[i,j].tolist() == [0, 0, 255] and coun == 0 :
                left_pred.append(j)
                coun += 1
            elif img_pred[i,j].tolist() == [0, 0, 255] and coun == 1 :
                right_pred.append(j)
        if coun == 0 :
            print(i)

    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        #y_true, y_pred = check_array(y_true.reshape(1351,1), y_pred.reshape(1351,1))

        ## Note: does not handle mix 1d representation
        #if _is_1d(y_true): 
        #    y_true, y_pred = _check_1d_array(y_true, y_pred)

        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(len(left_org),len(right_org))
    print(len(left_pred),len(right_pred))

    mse_left = mean_squared_error(left_pred, left_org)
    mse_right = mean_squared_error(right_pred, right_org)
    print(mse_left)
    print(mse_right)
    print(mean_absolute_percentage_error(left_org, left_pred))
    print(mean_absolute_percentage_error(right_org, right_pred))
    """ cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',img_pred)
    cv2.waitKey(0)
    cv2.destroyAllWindows() """

if __name__ == "__main__":
    imgs_mse('./data/img/result_imgs/201601.png','./data/img/result_imgs/label0.png')
    imgs_mse('./data/img/result_imgs/201701.png','./data/img/result_imgs/label1.png')
    pass