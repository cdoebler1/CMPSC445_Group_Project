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
with os.scandir(basedir) as faces:
    for face in faces:
        if face.is_file():
            img = load_img(basedir + face.name, color_mode = "grayscale")
            np_img = img_to_array(img)[:,:,0]
            i += 1
#        print(np.shape(np_img))
        np.savetxt(file,np_img)
print(str(i) + " images processed")
file.close()
