'''
Created on Oct 26, 2017

@author: kwibu
'''
"""
extract images as ndarray dataset into "./data" file.
"""
import cv2
import numpy as np
import glob, os

datalist = np.empty((0, 64, 64, 3), float)
dirs = ["./Aberdeen", "./pain", "./utrecht"]
for dir in dirs:
    os.chdir(dir)
    
    for image in glob.glob("*.jpg"):
        img = cv2.imread(image)
        border_width = max([0, int((img.shape[0]-img.shape[1])/2)])
        border_height = max([0, int((img.shape[1]-img.shape[0])/2)])
        bordered = cv2.copyMakeBorder(img, border_height, border_height, border_width, border_width,
                                     borderType = cv2.BORDER_REPLICATE)
        resized = cv2.resize(bordered, (64, 64))
        image_data = (np.array([resized])/255)*2-1
        datalist = np.vstack([datalist, image_data])
        
    data_file = open('./data', 'wb')
    np.save(data_file, datalist)
    data_file.close()
    os.chdir("../")
