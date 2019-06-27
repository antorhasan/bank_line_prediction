import numpy as np
import cv2

def auto_canny(image, sigma=0.99):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged
# Load an color image in grayscale
img = cv2.imread('../data/198901.tif', 1)   #can simply pass 1,0 or -1
#imgg = cv2.imread('../data/198901.tif', 0)
#high_thresh, thresh_im = cv2.threshold(imgg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#lowThresh = 0.5*high_thresh
denoise = cv2.bilateralFilter(img,9,75,75)
denoise = cv2.GaussianBlur(denoise,(5,5),0)
#img = cv2.Canny(img, 25, 250)
edges = auto_canny(denoise)
#contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img, contours, -1, (0,255,0), 3)

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
#cv2.imwrite('edges.jpg',edges)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

