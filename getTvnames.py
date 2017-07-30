#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np

import sys
import os

root = 'D:/work/All_tv_logo/'

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




logoDirs = getLogoDirs(root)
All_tv_names = []
imagePathsOfTvs = []
tv_name_num ={}

for logoDir in logoDirs:
    tv_Dir = getLogoDirs(logoDir)
    #print tv_Dir,'123*****'
    for dir in tv_Dir:

      imagePaths = readDir(dir)
      tv_num = len(imagePaths)
      #print len(imagePaths),'***'
      if len(imagePaths)>=100:
        name = os.path.basename(dir)
        #tvNames.append(name)
        #tvNames = getTvNames(dir)
        All_tv_names.append(name)
 
    #print name,tv_num
tvNames_file = open('tvNames.txt', 'w')
#for i,name in enumerate(tv_name_num):
    #result = '%s %d' % (name, tv_name_num[name])
    #tvNames_file.write(result+'\n')
    #print name,tv_name_num[name]

for name in All_tv_names:
    tvNames_file.write('\''+name+'\''+',')

#for i,name in enumerate(All_tv_names):
    #logo_name = All_tv_names[i]
    #logo_name.decode('utf-8').encode('gbk')
    #print i,All_tv_names[i]
    #All_tv_names[i]
    #print i,logo_name


#print len(All_tv_names)