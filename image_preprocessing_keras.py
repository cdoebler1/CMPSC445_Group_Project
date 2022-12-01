"""
Created on Thur Oct 20

@author: CMPSC445 WCFa22 Group 2
"""
import os
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import image_dataset_from_directory
import numpy as np
np.set_printoptions(threshold=np.inf)
import array

"""
i = 0
file = open("angry.csv", "w")
basedir = ('dataset/train/angry/')
with os.scandir(basedir) as faces:
    for face in faces:
       
        if face.is_file():
            img = load_img(basedir + face.name, color_mode = "grayscale")
            np_img = img_to_array(img)[:,:,0]
            i += 1
            temp = np_img.flatten()
            np.savetxt(file,np_img)
    print(np_img)
    print(temp)
 
print(str(i) + " images processed")
file.close()
"""
file = open("test.csv", "w")
basedir = ('dataset/train/')
#img = load_img(basedir + "im0.png", color_mode = "grayscale")
#np_img = img_to_array(img, dtype=int)[:,:,0].flatten()

np_img = image_dataset_from_directory(basedir, labels = 'inferred', label_mode = 'int',
                                      class_names=None, color_mode = "grayscale",
                                      batch_size=32, image_size=(48,48), shuffle=True,
                                      seed=None, validation_split=None, subset=None,
                                      interpolation='bilinear', follow_links=False,
                                      crop_to_aspect_ratio=False)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in np_img.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
        
#np.savetxt(file, np_img)
file.close()
