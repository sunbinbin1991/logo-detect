#!/usr/bin/env python2.7
# coding: utf-8


import cv2
import numpy as np

import sys
import os
import random
import math

from config_5_5 import bin_n, wc, hc, shape0, size0, width_percent, height_percent




def cut(img, width_percent, height_percent):
  height = img.shape[0]
  width = img.shape[1]
  width_cut = int(width*width_percent)
  height_cut = int(height*height_percent)
  cut_img = img[height_cut:(height-height_cut), width_cut:(width-width_cut)]
  return cut_img


def hog(img):
  h, w = img.shape

  gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
  gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)

  mag, ang = cv2.cartToPolar(gx, gy)
  bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)


  bin_cells = ()
  mag_cells = ()
  for i in range(wc):
    for j in range(hc):
      bin_cells += (bins[j*h/hc:(j+1)*h/hc, i*w/wc:(i+1)*w/wc],)
      mag_cells += (mag[j*h/hc:(j+1)*h/hc, i*w/wc:(i+1)*w/wc],)


  hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
  hist = np.hstack(hists)     # hist is a 16*wc*hc vector

  return hist


def resize_cut_hog_add_shape(img):
  shape = img.shape
  if shape != shape0:
    img=cv2.resize(img, size0, interpolation=cv2.INTER_LINEAR)
  img = cut(img, width_percent, height_percent)
  hist = hog(img)

  hist = np.concatenate((hist, shape))

  return hist 









