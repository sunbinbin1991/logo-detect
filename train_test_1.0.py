#!/usr/bin/env python2.7
# coding: utf-8


import cv2
import numpy as np

import sys
import os
import random
import math
import time

from config_5_4 import root, bin_n, feature_size, width_percent, height_percent, shape0, size0, tv_names, train_percent
from hog_5_4 import cut, hog, resize_cut_hog_add_shape
#import use_classifier_5_3 as c1
#import config_5_3 as config

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




def classify_logo(imagePath):

    predicted_label = 0
    img = cv2.imread(imagePath, 0)
    hist = np.float32(resize_cut_hog_add_shape(img))

    predicted_label = int(svm.predict(hist))

    #print 'predicted tv: ', predicted_tvName

    return predicted_label






# get all Images and all tvnames

logoDirs = getLogoDirs(root)

#tvNames = getTvNames(logoDirs)

imagePathsOfTvs = []
All_tv_names=[]

for logoDir in logoDirs:
    tv_Dir = getLogoDirs(logoDir)
    #print tv_Dir,'123*****'
    for dir in tv_Dir:

      imagePaths = readDir(dir)
      #print len(imagePaths),'***'
      if len(imagePaths)>100:
        name = os.path.basename(dir)
        #tvNames.append(name)
        imagePathsOfTvs.append(imagePaths)
        #tvNames = getTvNames(dir)
        All_tv_names.append(name)
    print logoDir

print 'get train and test images'

train_images = []
train_labels = []
test_images = []
test_labels = []



#  start  train


for i, paths in enumerate(imagePathsOfTvs):
  #paths.decode('utf-8').encode('gbk')
  random.shuffle(paths)
  n = len(paths)
  n_test = int(n*(1.0-train_percent))
  n_train = n - n_test
  #print len(paths)
  #if len(paths)>10:
    #random.shuffle(paths)
  

  if n_train>1000:
      n_train = 1000
  if n==1:
    n_test=1   
    train_images.extend(paths[:])
    train_labels.extend([i]*n_train)
    test_images.extend(paths[:])
    test_labels.extend([i]*n_test)
  else:
    if (n_test >1000):
        n_test = 1000
        train_images.extend(paths[:n_train])
        train_labels.extend([i]*n_train)
        test_images.extend(paths[n_train:n_train+n_test])
        test_labels.extend([i]*n_test)
    else:
        train_images.extend(paths[:n_train])
        train_labels.extend([i]*n_train)
        test_images.extend(paths[n_train:n_train+n_test])
        test_labels.extend([i]*n_test)
  #print i, 'n_train = ', n_train, 'n_test = ', n_test


print '*********************************************************************start training***************************************************************************'


train_labels = np.array(train_labels)
test_labels = np.array(test_labels)


n_train = len(train_images)
n_test = len(test_images)

train_hists = np.float32(np.zeros((n_train, feature_size))) 
#test_hists = np.float32(np.zeros((n_test,64)))

for i, imagePath in enumerate(train_images):
  if(i%1000==0):
    print imagePath
  img = cv2.imread(imagePath, 0)
  #cv2.imshow('img',img)
  hist = resize_cut_hog_add_shape(img)
  train_hists[i,:] = hist

#'''
svm_params = dict(kernel_type=cv2.SVM_LINEAR,
                  svm_type=cv2.SVM_C_SVC,
                  C=2.67, gamma=5.383)
svm = cv2.SVM()

print 'training svm'
print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
svm.train(train_hists, train_labels, params=svm_params)

print 'saving svm'
print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
svm.save('svm_5_3_1.0.xml')
time.sleep(10)
#'''
#the num of predict tv_logo right

print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))


# star test

print  '*****************************************************************************start test***********************************************************************'
svm_file = './svm_5_3_1.0.xml'

svm = cv2.SVM()
svm.load(svm_file)

test_right_logo ={}
test_wrong_logo = {}
# for each tv calculate numbers,统计每个台标数
test_tv_Nums ={}
for i, name in enumerate(tv_names):
    test_wrong_logo[name] = 0
    test_right_logo[name] = 0
    test_tv_Nums[name] = 0

# for each tv VS tv_labels
for i,name in enumerate(test_labels):
    #print test_labels[i]
    test_tv_Nums[tv_names[test_labels[i]]]  += 1
#print len(test_labels)
#for i,name in enumerate(test_tv_Nums):
    #print i,tv_names[i],test_tv_Nums[tv_names[i]]




for i,name in enumerate(test_images):
    imagePath = test_images[i]
    predictLabel =classify_logo(imagePath)
    if(predictLabel!=test_labels[i]):
        test_wrong_logo[tv_names[predictLabel]] += 1
        #print i, tv_names[predictLabel], tv_names[test_labels[i]], test_images[i], '**'
    else:
        test_right_logo[tv_names[predictLabel]] += 1
hit_rate_sum = 0
acc_rate_sum = 0

file_output = open('test_result.txt', 'a')
for i, name in enumerate(tv_names):
    hit_rate = 100.0*test_right_logo[tv_names[i]]/test_tv_Nums[tv_names[i]]
    if((test_right_logo[tv_names[i]]+test_wrong_logo[tv_names[i]])==0):
        accuracy = 100.0*test_right_logo[tv_names[i]]/1
    else:
        accuracy = 100.0*test_right_logo[tv_names[i]]/(test_right_logo[tv_names[i]]+test_wrong_logo[tv_names[i]])
    if (tv_names[i] != 'Notv'):
        hit_rate_sum += hit_rate
        acc_rate_sum += accuracy
    tv_names[i] = tv_names[i].decode('utf-8').encode('gbk')
    print tv_names[i] , 'hit rate = ',hit_rate,'%','accuracy = ',accuracy,'%'
    result = '%s: hit rate = %f, accuracy = %f' % (tv_names[i], hit_rate, accuracy)
    file_output.write(result + '\n')
hit_rate = 1.0 * hit_rate_sum / (len(tv_names)-1)
accuracy = 1.0 * acc_rate_sum / (len(tv_names)-1)
tot_result ='%s: hit_rate_sum = %f, acc_rate_sum = %f' % (tv_names[i], hit_rate, accuracy)

print 'hit_rate_sum = ', 1.0 * hit_rate_sum / (len(tv_names)-1) , '%'
print 'acc_rate_sum = ', 1.0 * acc_rate_sum / (len(tv_names)-1) , '%'
file_output.write(tot_result + '\n')



#'''


