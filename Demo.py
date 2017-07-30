# coding: utf-8
import os
import sys
import cv2

import use_classifier_5_3 as c1
import use_classifier_5_4 as c2

from tvs import tv_names

root =   './test_images/'
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

imagePaths = readDir(root)
for i ,path in enumerate (imagePaths):
    #print i,path
    predict_label_1 = c1.classify_logo(path)
    predict_label_2 = c2.classify_logo(path)


    if(predict_label_1 ==predict_label_2):
        su = u'tv_names[predict_label_1]'
        print tv_names[predict_label_1].decode('utf-8').encode('gb2312')
        #print type(tv_names[predict_label_1])
    else:
        print tv_names[0].decode('utf-8').encode('gb2312'),tv_names[predict_label_1].decode('utf-8').encode('gb2312'),'****',tv_names[predict_label_2].decode('utf-8').encode('gb2312')

    image = cv2.imread(path)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


