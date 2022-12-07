#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 13:48:15 2022

@author: chanjoelle
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img



train_dir = '../dataset/train'
test_dir = '../dataset/test'

imag_gen = ImageDataGenerator()

#img = load_img(train_dir + "/angry/" + os.listdir(train_dir + "/angry")[0])

train_set = imag_gen.flow_from_directory(train_dir) # train data generator

test_set = imag_gen.flow_from_directory(test_dir) #test data generator