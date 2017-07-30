#!/usr/bin/env python2.7
# coding: utf-8


import cv2
import numpy as np
import sys
import os
import random
import math
import time

import use_classifier_5_3 as c1
import use_classifier_5_4 as c2
from config_5_4 import tv_names

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

  #noLogoDir = root + 'Notv'

  logoDirs = []
  for f in os.listdir(root):

    #print f
    #if f != 'Notv':
      newPath = os.path.join(root, f)
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


if __name__ == '__main__':
  test_images = []
  test_labels = []
  test_percent = 0.5
  root = 'D:/work/All_tv_logo/'

  logoDirs = getLogoDirs(root)
  imagePathsOfTvs = []
  All_tv_names = []

  for logoDir in logoDirs:
    tv_Dir = getLogoDirs(logoDir)
    # print tv_Dir,'123*****'
    for dir in tv_Dir:

      imagePaths = readDir(dir)
      # print len(imagePaths),'***'
      if len(imagePaths) > 100:
        name = os.path.basename(dir)
        # tvNames.append(name)
        imagePathsOfTvs.append(imagePaths)
        # tvNames = getTvNames(dir)
        All_tv_names.append(name)
    print logoDir

  for i, paths in enumerate(imagePathsOfTvs):
    # paths.decode('utf-8').encode('gbk')
    random.shuffle(paths)
    n = len(paths)
    n_test = int(n * test_percent)

    if (n_test > 1000):
      n_test = 1000
      test_images.extend(paths[: n_test])
      test_labels.extend([i] * n_test)
    else:
      test_images.extend(paths[:n_test])
      test_labels.extend([i] * n_test)
        # print i, 'n_train = ', n_train, 'n_test = ', n_test

  test_right_logo = {}
  test_wrong_logo = {}
  # for each tv calculate numbers,统计每个台标数
  test_tv_Nums = {}
  for i, name in enumerate(tv_names):
    test_wrong_logo[name] = 0
    test_right_logo[name] = 0
    test_tv_Nums[name] = 0

  # for each tv VS tv_labels
  for i, name in enumerate(test_labels):
    # print test_labels[i]
    test_tv_Nums[tv_names[test_labels[i]]] += 1
    # print len(test_labels)
    # for i,name in enumerate(test_tv_Nums):
    # print i,tv_names[i],test_tv_Nums[tv_names[i]]

  for i, name in enumerate(test_images):
    imagePath = test_images[i]

    if c1.classify_logo(imagePath)==c2.classify_logo(imagePath):
      predictLabel = c1.classify_logo(imagePath)
    else:
      predictLabel = 0
    if (predictLabel != test_labels[i]):
      test_wrong_logo[tv_names[predictLabel]] += 1
      # print i, tv_names[predictLabel], tv_names[test_labels[i]], test_images[i], '**'
    else:
      test_right_logo[tv_names[predictLabel]] += 1
  hit_rate_sum = 0
  acc_rate_sum = 0

  file_output = open('test_results.txt', 'a')
  for i, name in enumerate(tv_names):
    hit_rate = 100.0 * test_right_logo[tv_names[i]] / test_tv_Nums[tv_names[i]]
    if ((test_right_logo[tv_names[i]] + test_wrong_logo[tv_names[i]]) == 0):
      accuracy = 100.0 * test_right_logo[tv_names[i]] / 1
    else:
      accuracy = 100.0 * test_right_logo[tv_names[i]] / (test_right_logo[tv_names[i]] + test_wrong_logo[tv_names[i]])
    if (tv_names[i] != 'Notv'):
      hit_rate_sum += hit_rate
      acc_rate_sum += accuracy
    tv_names[i] = tv_names[i].decode('utf-8').encode('gbk')
    print tv_names[i], 'hit rate = ', hit_rate, '%', 'accuracy = ', accuracy, '%'
    result = '%s: hit rate = %f, accuracy = %f' % (tv_names[i], hit_rate, accuracy)
    file_output.write(result + '\n')
  hit_rate = 1.0 * hit_rate_sum / (len(tv_names) - 1)
  accuracy = 1.0 * acc_rate_sum / (len(tv_names) - 1)
  tot_result = '%s: hit_rate_sum = %f, acc_rate_sum = %f' % (tv_names[i], hit_rate, accuracy)

  print 'hit_rate_sum = ', 1.0 * hit_rate_sum / (len(tv_names) - 1), '%'
  print 'acc_rate_sum = ', 1.0 * acc_rate_sum / (len(tv_names) - 1), '%'
  file_output.write(tot_result + '\n')















