#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 16:47:56 2022

@author: CMPSC445 WCFa22 Group 2
"""

import preprocessor as pp
import tensorflow as tf

# Preprocess the data set
train_data = pp.preprocess('dataset/train')
test_data = pp.preprocess('dataset/test')

# Display a few sample images from the training and test data sets
num_images = 9
pp.image_sample_display(train_data, num_images)
pp.image_sample_display(test_data, num_images)

# Train the model
# num_classes = 7
num_classes = len(train_data.class_names)

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)])

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_data,
  validation_data=test_data,
  epochs=3)