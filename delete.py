import os
import cv2 as cv 
import numpy as np 


count = 0
for counter, file in enumerate(os.listdir('/Users/sinhasam/Documents/lua/Ganerator/images')):
	if not file.endswith('.JPEG'):
		continue
	filename = str(file)

	img = cv.imread(filename)


	if img.shape[0] != 224 or img.shape[1] != 224:
		count += 1
		# os.remove(file)
print(count)
