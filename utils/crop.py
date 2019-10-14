import cv2
import numpy as np
import os
import glob
from os import listdir
from os.path import isfile, join

def crop_to_roi(where, input_dir):
	'''crop the jamuna river sat image into equal reaches with 256*768 size
		input: paths
		output: cropped images'''
	
	path = './data/finaljan/'

	files = [f for f in listdir(path) if isfile(join(path, f))]
	#files = files[0:2]

	coor_list = [516, 491, 516, 450, 396, 355, 325, 277, 310, 400]

	for i in range(len(files)):
		print(files[i])
		img = cv2.imread(path + files[i])

		for k in range(len(coor_list)):
			num = coor_list[k]

			crop_img = img[256*k : 256*(k+1),num:num+768]
			cv2.imwrite("./data/alt_la/"+files[i].split('.')[0]+str(k)+".png", crop_img)

def stitch_imgs():
	'''stitch 3 consecutive images into 1 image'''
	input_dir = './data/result/'
	file_list = [f for f in listdir(input_dir) if isfile(join(input_dir,f))]

	'''sort the files according to interger values of filenames'''
	for i in range(len(file_list)):
		file_list[i] = int(file_list[i].split('.')[0])
	file_list.sort()
	#print(file_list)

	for i in range(0, len(file_list), 3):
		#print(file_list[i])
		img1 = cv2.imread(input_dir + str(file_list[i]) + '.png', cv2.IMREAD_GRAYSCALE)
		img2 = cv2.imread(input_dir + str(file_list[i+1]) + '.png', cv2.IMREAD_GRAYSCALE)
		img3 = cv2.imread(input_dir + str(file_list[i+2]) + '.png', cv2.IMREAD_GRAYSCALE)

		#print(img1.shape)

		stitched_img = np.concatenate([img1, img2, img3], axis=1)

		cv2.imwrite('./data/result/stitched/' + str(i)+ '.png', stitched_img)
