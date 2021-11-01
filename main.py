import os
import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
from tensorflow.keras.preprocessing.image import load_img,img_to_array

print('Python version: %s' % sys.version)
print('TensorFlow version: %s' % tf.__version__)
print('Keras version: %s' % tf.keras.__version__)

####################
### LOADING DATA ###
####################
print("Loading and preprocessing data...")
inps = os.listdir("./training_data_inputs")
labels = os.listdir("./training_data_labels")
data = set(inps) & set(labels)

for a in data:
	img_input = img_to_array(load_img("./training_data_inputs/" + a, color_mode='rgb'))
	img_label = img_to_array(load_img("./training_data_labels/" + a, color_mode='rgb'))
	print(a)
	print(type(img_input))
	print(img_input[0, 0])
	print(img_input.size)
	print(img_input.shape)



"""
# Model goes here
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(kernel_size=3, filters=16, activation='relu', input_shape=[IMG_SIZE,IMG_SIZE, 3]),


model.compile(optimizer='adadelta',
              loss='mean_squared_error',
              metrics=['mean_absolute_error', 
                       'mean_squared_logarithmic_error',
                       'mean_squared_error',
                       'logcosh'])
"""