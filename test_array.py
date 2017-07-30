#!/usr/bin/env python2.7
# encoding: utf-8

import numpy
import cv2
A = [1,2,3,4]
B = ([5,6,7,8],[7,8,9,0])
imgPath = '安徽卫视.jpg'
imgPath = imgPath.decode('utf-8').encode('gbk')
#str = imgPath2.encode('utf-8')
img = cv2.imread(imgPath,0)
shape = img.shape
C = numpy.concatenate((A,shape))
print len(A)
print C
