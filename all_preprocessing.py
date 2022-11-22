#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 22:20:46 2022

@author: chanjoelle
"""

import os
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import numpy as np

# create seperate folder for csv data
train_directory = "train"
test_directory = "test"
parent_dir = "dataset/dataset_csv"
train_path = os.path.join(parent_dir, train_directory)
test_path = os.path.join(parent_dir, test_directory)

try:
    os.makedirs(train_path, exist_ok = True)
except OSError as error:
    print("Train Directory already exists")

try:
    os.makedirs(test_path, exist_ok = True)
except OSError as error:
    print("Test Directory already exists")
    

emotions = ["angry", "digusted", "fearful", "happy", "neutral", "sad", "suprised"]

#change to the train directory to create the corresponding csv files
os.chdir(train_path)

#create the csv file for every emotions
for i in range(len(emotions)):
    file = open(emotions[i]+".csv", "w")

#change to the test directory to create the corresponding csv files
os.chdir(test_path)

#create the csv file for every emotions
for i in range(len(emotions)):
    file = open(emotions[i]+".csv", "w")

    
original_path = "dataset/train/angry/"

os.chdir(original_path)    
basedir = ('dataset/train/angry/')
with os.scandir(basedir) as faces:
    for face in faces:
        if face.is_file():
            img = load_img(basedir + face.name, color_mode = "grayscale")
            np_img = img_to_array(img)[:,:,0]
print(np.shape(np_img))
np.savetxt(file,np_img)
file.close()
