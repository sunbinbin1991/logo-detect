#!/usr/bin/env python2.7
# coding: utf-8


import cv2
import numpy as np

import sys
import os
import random
import math

from config import root, bin_n, feature_size, width_percent, height_percent, shape0, size0, tv_names, train_percent
from hog import cut, hog, resize_cut_hog_add_shape

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

  noLogoDir = dirPath + '无台标'

  logoDirs = []
  for f in os.listdir(dirPath):
    if f != '无台标':
      newPath = os.path.join(dirPath, f)
      if os.path.isdir(newPath):
        logoDirs.append(newPath)

  logoDirs.append(noLogoDir)

  return logoDirs


def getTvNames(logoDirs):

  tvNames = []
  for f in logoDirs:
    name = os.path.basename(f)
    tvNames.append(name)

  return tvNames




print 'fetching images ...'

logoDirs = getLogoDirs(root)

tvNames = getTvNames(logoDirs)

imagePathsOfTvs = []


for dir in logoDirs:

  imagePaths = readDir(dir)

  imagePathsOfTvs.append(imagePaths)


train_images = []
train_labels = []
test_images = []
test_labels = []


# train
print 'get train and test images'


for i, paths in enumerate(imagePathsOfTvs):

  random.shuffle(paths)
  n = len(paths)
  n_test = int(n*(1.0-train_percent))
  n_train = n - n_test
  print tv_names[i], 'n_train =', n_train, 'n_test =', n_test

  train_images.extend(paths[:n_train])
  train_labels.extend([i]*n_train)
  test_images.extend(paths[n_train:])
  test_labels.extend([i]*n_test)




print 'computing hists ...'

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

n_train = len(train_images)
n_test = len(test_images)

train_hists = np.float32(np.zeros((n_train, feature_size))) 

for i, imagePath in enumerate(train_images):
  img = cv2.imread(imagePath, 0)
  hist = resize_cut_hog_add_shape(img)
  train_hists[i,:] = hist




print 'training svm ...'
svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=2.67, gamma=5.383 )

svm = cv2.SVM()

svm.train(train_hists, train_labels, params=svm_params)

svm.save('svm.xml')










