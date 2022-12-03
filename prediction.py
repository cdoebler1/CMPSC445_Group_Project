#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 06:11:04 2022

@author: CMPSC445 WCFa22 Group 2
"""

import numpy as np
import preprocessor as pp
import tensorflow as tf
from tkinter import filedialog


def prediction(sample):
    # Load model
    model = tf.keras.models.load_model('./save', compile = True)
    class_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    sample_data = np.array(sample)
    sample_data = sample_data[np.newaxis,:,:]
    prediction = model.predict(sample_data)
    num_class = np.argmax(prediction, axis = 1)
    name_class = class_names[num_class[0]]
    
    return name_class


def main ():

    sample_path = filedialog.askopenfilename()
    sample = tf.keras.utils.load_img(sample_path, color_mode = "grayscale")
    name_class = prediction(sample)
    pp.single_image_display(sample, name_class)

if __name__ == '__main__':
    main()

