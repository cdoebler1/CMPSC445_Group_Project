#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 22:20:46 2022

@author: CMPSC445 WCFa22 Group 2 Test
"""

from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt

def preprocess(data_dir):
    data_set = image_dataset_from_directory(data_dir, labels = 'inferred', label_mode = 'int',
                                          class_names=None, color_mode = "grayscale",
                                          batch_size=32, image_size=(48,48), shuffle=True,
                                          seed=None, validation_split=None, subset=None,
                                          interpolation='bilinear', follow_links=False,
                                          crop_to_aspect_ratio=False)
    return(data_set)

def image_sample_display(data_set, num_images):
    class_names = data_set.class_names
    plt.figure(figsize=(10, 10))
    for images, labels in data_set.take(1):
        for i in range(num_images):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")


def single_image_display (sample, prediction):
    plt.figure(figsize=(10,10))
    plt.imshow(sample)
    plt.title(prediction)
               