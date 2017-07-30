#!/usr/bin/env python2.7
# coding: utf-8


import cv2
import numpy as np

import sys
import os
import math

from config import root, bin_n, width_percent, height_percent, shape0, size0, tv_names

from hog import cut, hog, resize_cut_hog_add_shape



svm_file = '../svm_5_3.xml'

svm = cv2.SVM()
svm.load(svm_file)



def classify_logo(imagePath):

    predicted_label = 0

    img = cv2.imread(imagePath, 0)
    hist = np.float32(resize_cut_hog_add_shape(img))

    predicted_label = int(svm.predict(hist))
    predicted_tv_name = tv_names[predicted_label]

    #print 'predicted tv: ', predicted_tvName

    return predicted_label










