'''
Created on Oct 26, 2017

@author: kwibu
'''
import cv2
import numpy as np
import glob, os

datalist = np.empty((0, 128, 128, 3), float)
os.chdir("./Myface")
import matplotlib.pyplot as plt

for image in glob.glob("*.jpg"):
    img = cv2.imread(image)
    img = img[:, int(img.shape[1]/2-img.shape[0]/2):int(img.shape[1]/2+img.shape[0]/2)]
    resized = cv2.resize(img, (128, 128))
    image_data = (np.array([resized])/255)*2-1
    datalist = np.vstack([datalist, image_data])
    
data_file = open('./data', 'wb')
np.save(data_file, datalist)
data_file.close()
