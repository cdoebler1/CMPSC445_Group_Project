#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 16:47:56 2022

@author: CMPSC445 WCFa22 Group 2
"""

from preprocessor import preprocess, image_sample_display
import tensorflow as tf
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Rescaling, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import load_model, save_model

import datetime

""" The following 2 lines of code are a fix for a problem with Anaconda/Spyder
installs. If packages are installed in a certain order, you may have 
duplicate libraries. I don't need this code on my home computer, but it is
necessary on my work computer. The real solution is to uninstall the packages
and then reinstall them in a magical sequence unknown to anyone. """
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Preprocess the data set
batch_size = 32
train_data = preprocess('dataset/train', batch_size)
validation_data = preprocess('dataset/test', batch_size)
num_classes = len(train_data.class_names)

# Set up a decreasing learning rate. Start decreasing at 10 epochs.
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
image_sample_display(train_data, num_images)
image_sample_display(validation_data, num_images)

# Define a few models

# Model1 trains in approximately 30 seconds per epoch
model1 = Sequential([
  Rescaling(1./255),
  Conv2D(32, 3, activation='relu'),
  MaxPooling2D(),
  Conv2D(64, 3, activation='relu'),
  MaxPooling2D(),
  Conv2D(64, 3, activation='relu'),
  MaxPooling2D(),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(num_classes)])

# Model2 trains in approximately 600 seconds per epoch
model2 = Sequential([

    Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape = (48, 48, 1)),
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)),
    Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    
    Dense(num_classes, activation='softmax')])

# Instantiate the desired model.
model = model1

# Train the selected model.
model.compile(
    Adam(learning_rate=0.0005), #default 0.001
    loss=SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

model.fit(
  train_data,
  validation_data=validation_data,
  epochs=50,
  callbacks=here_kitty_kitty)

# The checkpoint callback saves the best training as measured by val_accuracy.
# This step reloads the best saved model as the active model.
model = load_model('./checkpoint.h5', compile = True)

# Print the validation accuracy and loss of the best model.
score = model.evaluate(validation_data, verbose=0)
print(f'\nValidation accuracy: {score[1]}\nValidation loss: {score[0]}\n')

# Save model. This is a redundant function, since the best model is already
# saved as checkpoint.h5. 
save_model(model, './save')

