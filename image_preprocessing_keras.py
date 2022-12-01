"""
Created on Thur Oct 20

@author: CMPSC445 WCFa22 Group 2
"""
import os
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


i = 0
##file = open("angry.csv", "w")
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
dataframe = pd.DataFrame(all_data)
dataframe['emotion'] = 'angry'
with open('angry_!.csv','w') as file:
    dataframe.to_csv(path_or_buf= file, index = False, header = False)
pd.DataFrame.to_csv(dataframe)
#print(all_data)
print(str(i) + " images processed")
file.close()



"""