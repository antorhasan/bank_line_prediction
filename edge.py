import numpy as np
import cv2
from os import listdir
from os.path import isfile, join

path = './data/infra1/'

def auto_canny(image, sigma=0.45):
	
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged
# Load an color image in grayscale

files = [f for f in listdir(path) if isfile(join(path, f))]
#files = files[0:2]

for i in files:
    print(i)
    
    img = cv2.imread(path + i)     #can simply pass 1,0 or -1
    denoise = cv2.bilateralFilter(img,15,75,75)
    denoise = cv2.GaussianBlur(denoise,(17,11),0)
    #img = cv2.Canny(img, 25, 250)
    edges = auto_canny(denoise)
    cv2.threshold(img,250,255,cv2.THRESH_BINARY)
    cv2.imwrite('./data/label1/'+str(i).split('.')[0]+'.png',edges)

""" lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=1,maxLineGap=2)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2) """

""" lines = cv2.HoughLines(edges,1,np.pi/180,150)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2) """

#cv2.imwrite('houghlines5.jpg',img)

""" cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', edges)
cv2.waitKey(0)
cv2.destroyAllWindows() """

