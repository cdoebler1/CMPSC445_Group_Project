"""
Created on Thur Oct 20

@author: CMPSC445 WCFa22 Group 2
"""
import os
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import numpy as np


i = 0
file = open("angry.csv", "w")
basedir = ('dataset/train/angry/')
all_data = []
with os.scandir(basedir) as faces:
    for face in faces:
       
        if face.is_file():
            img = load_img(basedir + face.name, color_mode = "grayscale")
            np_img = img_to_array(img)[:,:,0]
            i += 1
            temp = np_img.flatten()# 2D to 1D
            all_data.extend(temp)
           
all_data = np.reshape(all_data,(-1,2304))
np.savetxt(file,all_data)
print(all_data)
print(str(i) + " images processed")
file.close()
