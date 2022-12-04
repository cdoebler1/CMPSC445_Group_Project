#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 16:47:56 2022

@author: CMPSC445 WCFa22 Group 2
"""

import preprocessor as pp
import tensorflow as tf
import datetime

# Preprocess the data set
train_data = pp.preprocess('dataset/train')
test_data = pp.preprocess('dataset/test')

# Define the Keras TensorBoard callback.
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                      histogram_freq=1)

# Display a few sample images from the training and test data sets
num_images = 9
pp.image_sample_display(train_data, num_images)
pp.image_sample_display(test_data, num_images)

# Train the model
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
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), #default 0.001
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_data,
  validation_data=test_data,
  epochs=50,
  callbacks=[tensorboard_callback])

score = model.evaluate(test_data, verbose=0)
print(f'\nTest loss: {score[0]} / Test accuracy: {score[1]}\n')

# Save model
tf.keras.models.save_model(model, './save')

