import os
import time

from django.shortcuts import render, redirect
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage

import numpy as np

import prediction
import preprocessor as pp
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import datetime
import time


def result(request):
    return render(request, 'interface/result.html')


def error(request):
    return render(request, 'interface/error.html')


def train(request):
    # Preprocess the data set
    train_data = pp.preprocess('ML/dataset/train')
    validation_data = pp.preprocess('ML/dataset/test')

    # Define the Keras TensorBoard callback.
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    here_kitty_kitty = [TensorBoard(log_dir=log_dir, histogram_freq=1),
                        EarlyStopping(monitor='val_accuracy', patience=5),
                        ModelCheckpoint(filepath='best.h5', monitor='val_accuracy',
                                        save_best_only=True)]

    # Display a few sample images from the training and test data sets
    num_images = 9
    pp.image_sample_display(train_data, num_images)
    pp.image_sample_display(validation_data, num_images)

    # Train the model
    num_classes = len(train_data.class_names)

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # default 0.001
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=50,
        callbacks=here_kitty_kitty)

    model = tf.keras.models.load_model('./best.h5', compile=True)

    score = model.evaluate(validation_data, verbose=0)
    print(f'\nTest loss: {score[0]} / Test accuracy: {score[1]}\n')

    # Save model
    tf.keras.models.save_model(model, './save')
    return render(request, 'interface/train.html')


def home(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)

        base = settings.MEDIA_ROOT
        filepath = os.path.join(base, str(filename))

        try:
            start = time.time()
            image = tf.keras.utils.load_img(filepath, color_mode="grayscale")
            predictionCall = prediction.prediction(image)
            end = time.time()
            elapsed = end - start
        except OSError:
            return render(request, 'interface/error.html', {
                'ERROR': "Model not Generated",
            })
        except:
            return render(request, 'interface/error.html', {
                'ERROR': "Unable to process image, please try another.",
            })

        return render(request, 'interface/result.html', {
            'uploaded_file_url': uploaded_file_url,
            'prediction': predictionCall,
            'runtime': elapsed
        })
    return render(request, 'interface/home.html')
