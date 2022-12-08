#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 16:47:56 2022

@author: CMPSC445 WCFa22 Group 2
"""

import preprocessor as pp
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
import datetime

"""# The following 2 lines of code are a fix for a problem with Anaconda/Spyder
installs. If packages are installed in a certain orders, you may have 
duplicate libraries. I don't need this code on one computer that I use, but it is
necessary on my work computer. The real solution is to uninstall the packages
and then reinstall them in a magical sequence unknown to anyone. """
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Preprocess the data set
batch_size = 32
train_data = pp.preprocess('dataset/train', batch_size)
validation_data = pp.preprocess('dataset/test', batch_size)

# Set up a decreasing learning rate Start decreasing at 10 epochs.
def scheduler (epoch, learning_rate):
    if epoch < 10:
        return learning_rate
    else:
        return learning_rate * tf.math.exp(-0.1)

# Define the Keras TensorBoard callback.
buried_bodies = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
here_kitty_kitty = [TensorBoard(log_dir=buried_bodies, histogram_freq=1),
                    EarlyStopping(monitor='val_accuracy', patience=5),
                    ModelCheckpoint(filepath='checkpoint.h5',
                                    monitor='val_accuracy',
                                    save_best_only=True),
                    LearningRateScheduler(scheduler, verbose=0)]

# Display a few sample images from the training and test data sets
num_images = 9
pp.image_sample_display(train_data, num_images)
pp.image_sample_display(validation_data, num_images)

# Train the model
num_classes = len(train_data.class_names)

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, activation='relu'),
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
  validation_data=validation_data,
  epochs=50,
  callbacks=here_kitty_kitty)

model = tf.keras.models.load_model('./checkpoint.h5', compile = True)

score = model.evaluate(validation_data, verbose=0)
print(f'\nTest loss: {score[0]} / Test accuracy: {score[1]}\n')

# Save model
tf.keras.models.save_model(model, './save')

