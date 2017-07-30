#!/usr/bin/env python2.7
# coding: utf-8


import cv2
import numpy as np
import sys
import os
import random
import math
import time

from deploy import classify_logo
import config



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


def getLogoDirs(root):

  noLogoDir = root + 'Notv'

  logoDirs = []
  for f in os.listdir(root):
    if f != 'Notv':
      newPath = os.path.join(root, f)
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




root = config.root

logoDirs = getLogoDirs(root)

tvNames = config.tv_names


n_tv = len(logoDirs)

n_test_tv = [0]*n_tv
n_correct_tv = [0]*n_tv # n_correct_tv[i]为被正确判为i的图片数
n_wrong_tv = [0]*n_tv # n_wrong_tv[i]为被误判为i的图片数
hit_rate_tv = [0]*n_tv # hit_rate = n_correct/n_test
accuracy_tv = [0]*n_tv # accuracy = n_correct/(n_correct+n_wrong)

for id, dir in enumerate(logoDirs):

  n_test = config.n_test

  print id

  testImagePaths = []
  imagePaths = readDir(dir)
  random.shuffle(imagePaths)

  n = len(imagePaths)

  if n < n_test:
    n_test = n

  testImagePaths = imagePaths[:n_test]

  n_test_tv[id] = n_test

  for imagePath in testImagePaths:
    predicted_label = classify_logo(imagePath)
    if predicted_label == id:
      n_correct_tv[id] += 1
    else:
      n_wrong_tv[predicted_label] += 1





file_output = open('test_result.txt', 'a')

for i in range(n_tv):
  tv_name = tvNames[i]
  n_test = n_test_tv[i]
  n_correct = n_correct_tv[i]
  n_wrong = n_wrong_tv[i]

  hit_rate = 1.0*n_correct/n_test
  accuracy = 1.0*n_correct/(n_correct+n_wrong)

  result = '%s: hit rate = %f, accuracy = %f' % (tv_name, hit_rate, accuracy)
  print result
  file_output.write(result+'\n')

  hit_rate_tv[i] = hit_rate
  accuracy_tv[i] = accuracy





















