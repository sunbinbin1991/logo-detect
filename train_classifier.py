#!/usr/bin/env python2.7
# coding: utf-8


import cv2
import numpy as np

import sys
import os
import random
import math
import time

from config_5_3 import root, bin_n, feature_size, width_percent, height_percent, shape0, size0, tv_names, train_percent
from hog_5_3 import cut, hog, resize_cut_hog_add_shape

reload(sys)  
sys.setdefaultencoding('utf8')



def readDir(filePath):
    fileNames = []
    if os.path.isdir(filePath):
        for f in os.listdir(filePath):
            newFilePath = os.path.join(filePath, f)
            if os.path.isdir(newFilePath):
                fileNames.extend(readDir(newFilePath))
            elif os.path.splitext(f)[-1] == '.JPG' or os.path.splitext(f)[-1] == '.jpg':
                fileNames.append(newFilePath)
        return fileNames
    else:
        return filePath



def readDirWithTargetShape(filePath, targetShape):
    fileNames = []
    if os.path.isdir(filePath):
        for f in os.listdir(filePath):
            newFilePath = os.path.join(filePath, f)
            if os.path.isdir(newFilePath):
                fileNames.extend(readDir(newFilePath))
            elif os.path.splitext(f)[-1] == '.JPG' or os.path.splitext(f)[-1] == '.jpg':
                img = cv2.imread(newFilePath,0)
                if img.shape == targetShape:
                  fileNames.append(newFilePath)
        return fileNames
    else:
        return filePath



def show_cut_img(img_name):
  img = cv2.imread(img_name, 0)

  cut_img = cut(img)

  cv2.imshow('cut image', cut_img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  return cut_img



# 得到所有包含图标的文件夹地址，台标的id为在logoDirs中的位置

def getLogoDirs(dirPath):

  #noLogoDir = dirPath + 'Notv'

  logoDirs = []
  for f in os.listdir(dirPath):
    #if f != 'Notv':
      newPath = os.path.join(dirPath, f)
      if os.path.isdir(newPath):
        logoDirs.append(newPath)

  #logoDirs.append(noLogoDir)

  return logoDirs


def getTvNames(logoDirs):

  tvNames = []
  for f in logoDirs:
    name = os.path.basename(f)
    tvNames.append(name)

  return tvNames


logoDirs = getLogoDirs(root)

#tvNames = getTvNames(logoDirs)

imagePathsOfTvs = []


All_tv_names=[]

for logoDir in logoDirs:
    tv_Dir = getLogoDirs(logoDir)
    tvNames = getTvNames(tv_Dir)
    
    for dir in tv_Dir:

      imagePaths = readDir(dir)
      #print len(imagePath)

      imagePathsOfTvs.append(imagePaths)

    All_tv_names.extend(tvNames)
    #print logoDir











train_images = []
train_labels = []
test_images = []
test_labels = []


# train
print 'get train and test images'


for i, paths in enumerate(imagePathsOfTvs):
  #paths.decode('utf-8').encode('gbk')
  n = len(paths)
  n_test = int(n*(1.0-train_percent))
  n_train = n - n_test
  #print len(paths)
  #if len(paths)>10:
    #random.shuffle(paths)
  

  if n_train>200:
      n_train = 200
  


  if n==1:
    n_test=1
    print n_train
    
    train_images.extend(paths[:])

    train_labels.extend([i]*n_train)
    test_images.extend(paths[:])
    test_labels.extend([i]*n_test)
  else:
    train_images.extend(paths[:n_train])
    train_labels.extend([i]*n_train)
    test_images.extend(paths[n_train:])
    test_labels.extend([i]*n_test)
  print i, 'n_train = ', n_train, 'n_test = ', n_test
print 'get hists'


train_labels = np.array(train_labels)
test_labels = np.array(test_labels)


n_train = len(train_images)
n_test = len(test_images)

train_hists = np.float32(np.zeros((n_train, feature_size))) 
#test_hists = np.float32(np.zeros((n_test,64)))

for i, imagePath in enumerate(train_images):
  #print imagePath,'****'
  img = cv2.imread(imagePath, 0)
  #cv2.imshow('img',img)
  hist = resize_cut_hog_add_shape(img)
  train_hists[i,:] = hist

'''
for i, imagePath in enumerate(test_images):
  img = cv2.imread(imagePath, 0)

  img = cut(img, width_percent, height_percent)

  hist = hog(img)
  test_hists[i,:] = hist
'''

#svm_params = dict( kernel_type = cv2.SVM_LINEAR,
#                    svm_type = cv2.SVM_C_SVC,
 #                   C=2.67, gamma=5.383 )

svm_params = dict(kernel_type=cv2.SVM_LINEAR,
                  svm_type=cv2.SVM_C_SVC,
                  C=2.67, gamma=5.383)
svm = cv2.SVM()

print 'training svm'
print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
svm.train(train_hists, train_labels, params=svm_params)

print 'saving svm'
print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
svm.save('svm_5_3_0.xml')










