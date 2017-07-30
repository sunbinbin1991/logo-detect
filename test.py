#!/usr/bin/env python2.7
# encoding: utf-8


import cv2
import numpy as np
import sys
import os
import config_5_3 as config
tvnames= config.tv_names
#for Name in tvnames:
    #print  Name

#imgPath =tvnames[24]+'.jpg'

#imgPath ='\xe5\xae\x89\xe5\xbe\xbd\xe5\x8d\xab\xe8\xa7\x86' + '.jpg'
imgPath = '安徽卫视.jpg'
#print imgPath
#
print imgPath
imgPath = imgPath.decode('utf-8').encode('gbk')
#str = imgPath2.encode('utf-8')
img = cv2.imread(imgPath,0)
#imgPath = imgPath.decode('gbk')
#imgPath.encode('utf-8')
#print imgPath2
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()