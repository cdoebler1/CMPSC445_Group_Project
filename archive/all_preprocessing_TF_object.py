#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 22:20:46 2022
Update on Sat Nov 26 cdoebler

@author: chanjoelle
"""

from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt

train_dir = ('dataset/train/')
train_img = image_dataset_from_directory(train_dir, labels = 'inferred', label_mode = 'int',
                                      class_names=None, color_mode = "grayscale",
                                      batch_size=32, image_size=(48,48), shuffle=True,
                                      seed=None, validation_split=None, subset=None,
                                      interpolation='bilinear', follow_links=False,
                                      crop_to_aspect_ratio=False)

test_dir = ('dataset/test/')
test_img = image_dataset_from_directory(test_dir, labels = 'inferred', label_mode = 'int',
                                      class_names=None, color_mode = "grayscale",
                                      batch_size=32, image_size=(48,48), shuffle=True,
                                      seed=None, validation_split=None, subset=None,
                                      interpolation='bilinear', follow_links=False,
                                      crop_to_aspect_ratio=False)


plt.figure(figsize=(10, 10))
for images, labels in train_img.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")

plt.figure(figsize=(10, 10))
for images, labels in test_img.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")