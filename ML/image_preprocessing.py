"""
Created on Thur Oct 20

@author: CMPSC445 WCFa22 Group 2
"""
from PIL import Image
from numpy import asarray

img = Image.open('../dataset/train/angry/im0.png')
data = asarray(img)

print(data)
