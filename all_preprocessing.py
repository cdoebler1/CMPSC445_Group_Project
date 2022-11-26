#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 22:20:46 2022
Update on Sat Nov 26 cdoebler

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
    
emotions = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

#create the csv file for every emotions
for i in range(len(emotions)):
    train_path_emotion = os.path.join(train_path, emotions[i])
    train_data_emotion = os.path.join("dataset/train", emotions[i])
    test_path_emotion = os.path.join(test_path, emotions[i])
    test_data_emotion = os.path.join("dataset/test", emotions[i])
    file = open(train_path_emotion + ".csv", "w")
    with os.scandir(train_data_emotion) as faces:
        for face in faces:
             if face.is_file():
                img = load_img(os.path.join(train_data_emotion, face.name), color_mode = "grayscale")
                np_img = img_to_array(img)[:,:,0]
                np.savetxt(file, np_img)
    file.close()
    file = open(test_path_emotion + ".csv", "w")
    with os.scandir(test_data_emotion) as faces:
        for face in faces:
            if face.is_file():
                img = load_img(os.path.join(test_data_emotion, face.name), color_mode = "grayscale")
                np_img = img_to_array(img)[:,:,0]
                np.savetxt(file, np_img)
    file.close()
