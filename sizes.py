import cv2 
import os
import numpy as np

lowestHeight = 0
highestHeight = 0
lowestWidth = 0
highestWidth = 0
height = 0
total = 0
width = 0
for file in os.listdir('/Users/sinhasam/Documents/lua/Ganerator/images'):
	if not file.endswith('.JPEG'):
		continue
	filename = str(file)
	imgArray = cv2.imread(filename)
	total += 1
	if imgArray.shape[0] <= 450:
		height += 1
	if imgArray.shape[1] <= 500:
		width += 1
	# if imgArray.shape[0] < lowestHeight:
	# 	lowestHeight = imgArray.shape[0]
	# elif imgArray.shape[0] > highestHeight:
	# 	highestHeight = imgArray.shape[0]
	# if imgArray.shape[1] < lowestWidth:
	# 	lowestWidth = imgArray.shape[1]
	# elif imgArray.shape[1] > highestWidth:
	# 	highestWidth = imgArray.shape[1]


# print('lowest h: ' + str(lowestHeight))
# print('highest h: ' + str(highestHeight))
# print('lowest w' + str(lowestWidth))
# print('highest w' + str(highestWidth))
print('total ', total)
print('height ', height)
print('width ', width)