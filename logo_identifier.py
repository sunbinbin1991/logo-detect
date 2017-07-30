#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2
import numpy as np

import sys
import os
import random
import math
import datetime
import sys  


from tvs import tv_names
from test_classifier import classify_logo


reload(sys)  
sys.setdefaultencoding('utf8')


def read_dir(file_path):
    file_names = []
    if os.path.isdir(file_path):
        for f in os.listdir(file_path):
            newfile_path = os.path.join(file_path, f)
            if os.path.isdir(newfile_path):
                file_names.extend(read_dir(newfile_path))
            elif os.path.splitext(f)[-1].lower() == '.jpg':
                file_names.append(newfile_path)
        return file_names
    else:
        return file_path



# given a group of images to identify the logos
def identify_logos(images_dir):
  results = []
  test_images = read_dir(images_dir)
  for image_path in test_images:
    predicted_id = classify_logo(image_path)
    predicted_tv = tv_names[predicted_id]
    result = {'image': image_path, 'result': predicted_tv}
    results.append(result)
  return results


