import cv2 
import os
import numpy as np
from skimage.measure import block_reduce

HEIGHT = 224
WIDTH = 224




for count, file in enumerate(os.listdir('/Users/sinhasam/Documents/lua/Ganerator/images')):
	finalImage = np.zeros((HEIGHT, WIDTH, 3))
	if not file.endswith('.JPEG'):
		continue
	
	filename = str(file)
	imgArray = cv2.imread(filename)

	imgArray = block_reduce(imgArray, block_size = (2,2,1), func = np.mean)
	
	tempImg = np.zeros((HEIGHT, imgArray.shape[1], 3), dtype = np.int)
	imageHeight = imgArray.shape[0]
	imageWidth = imgArray.shape[1]
	if imageHeight > HEIGHT:
		if imageWidth > WIDTH:
			finalImage = imgArray[:HEIGHT, :WIDTH, :]
		elif imageWidth < WIDTH:
			finalImage[:HEIGHT, :imageWidth, :] = imgArray[:HEIGHT, :imageWidth, :]
		else:
			finalImage = imgArray

		# tempImg = imgArray[diffHeightTop : imgArray.shape[0] - diffHeightBottom + 3][:][:] 
	
	elif imageHeight < HEIGHT:
		if imageWidth > WIDTH:
			finalImage[:imageHeight, : WIDTH, :] = imgArray[:imageHeight, :WIDTH, :]
		elif imageWidth < WIDTH:
			finalImage[:imageHeight, :imageWidth,:] = imgArray
		else:
			finalImage = imgArray
	else:
		finalImage = imgArray


	print(count)
	
	cv2.imwrite(filename, finalImage)
	
	