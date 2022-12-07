import os
import time

from django.shortcuts import render, redirect
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage

import numpy as np

import ML.prediction
import ML.preprocessor as pp
import tensorflow as tf
import keras


# def home(request):
#    return render(request, 'interface/home.html')


def result(request):
    return render(request, 'interface/result.html')


def error(request):
    return render(request, 'interface/error.html')


def train(request):
    # Preprocess the data set
    train_data = pp.preprocess('ML/dataset/train')
    test_data = pp.preprocess('ML/dataset/test')

    # Display a few sample images from the training and test data sets
    num_images = 9
    pp.image_sample_display(train_data, num_images)
    pp.image_sample_display(test_data, num_images)

    # Train the model
    # num_classes = 7
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
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    model.fit(
        train_data,
        validation_data=test_data,
        epochs=3)

    score = model.evaluate(test_data, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

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
            image = tf.keras.utils.load_img(filepath, color_mode="grayscale")
            prediction = ML.prediction.prediction(image)
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
            'prediction': prediction
        })
    return render(request, 'interface/home.html')
