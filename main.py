import os
import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import time
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

koppens = [
	[255, 255, 255],
	[0, 0, 255],
	[0, 120, 255],
	[70, 170, 250],
	[255, 0, 0],
	[255, 150, 150],
	[245, 165, 0],
	[255, 220, 100],
	[255, 255, 0],
	[200, 200, 0],
	[150, 150, 0],
	[150, 255, 150],
	[100, 200, 100],
	[50, 150, 50],
	[200, 255, 80],
	[100, 255, 80],
	[50, 200, 0],
	[255, 0, 255],
	[200, 0, 200],
	[150, 50, 150],
	[170, 175, 255],
	[89, 120, 220],
	[75, 80, 179],
	[0, 255, 255],
	[55, 200, 255],
	[0, 125, 125],
	[178, 178, 178],
	[102, 102, 102]
]

x_train = []
y_train = []

for a in data:
	start_time = time.time()

	img_input = img_to_array(load_img("./training_data_inputs/" + a, color_mode='rgb'))
	img_label = img_to_array(load_img("./training_data_labels/" + a, color_mode='rgb'))

	input_data = np.zeros((img_input.shape[0], img_input.shape[1], 6))
	label_data = np.zeros((img_input.shape[0], img_input.shape[1], 28))

	for y in range(0, img_input.shape[0]):
		for x in range(0, img_input.shape[1]):
			# Process input
			p = img_input[y, x]
			if all(p == [0, 0, 255]):
				input_data[y, x, 0] = 1 # sea
			elif all(p == [177, 216, 230]):
				input_data[y, x, 1] = 1 # shelf
			elif all(p == [0, 0, 139]):
				input_data[y, x, 2] # trench
			elif all(p == [0, 255, 0]):
				input_data[y, x, 3] # plains
			elif all(p == [150, 75, 0]):
				input_data[y, x, 4] # mountains
			elif all(p == [112, 128, 144]):
				input_data[y, x, 5] # tall mountains
			else:
				raise Exception("UNKNOWN INPUT COLOR IN : " + a) # unknown
			# Process label
			l = img_label[y, x]
			min_dist = 255 * 4
			index = 0
			for n in range(0, len(koppens)):
				h = koppens[n]
				dist = abs(h[0] - l[0]) + abs(h[1] - l[1]) + abs(h[2] - l[2])
				if dist < min_dist:
					min_dist = dist
					index = n
					if dist < 5:
						break
			if min_dist > 5:
				raise Exception("NO PIXEL SEEMS TO BE A CLOSE FIT FOR PIXEL: " + str(x) + ", " + str(y) + " IN: " + str(a) + " WITH COLOR: " + str(l))
			label_data[y, x, index] = 1

	x_train.append(input_data)
	y_train.append(label_data)

	end_time = time.time()
	print(str(a) + ": " + str(end_time - start_time) + "s")


print("Image loaded!")

x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train[0].shape)
print(y_train[0].shape)


model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(400, 800, 6,)))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(len(koppens), kernel_size=(3, 3), activation='relu'))
#model.add(tf.keras.layers.Flatten())
#model.add(tf.keras.layers.Dense(1))
#model.add(tf.keras.layers.Flatten())
#model.add(tf.keras.layers.Dense(1, activation='relu'))
#model.add(tf.keras.layers.Dense(400*800*len(koppens)))
#model.add(tf.keras.layers.Reshape((400, 800, len(koppens),)))

print("--- compiling the model ---")
model.compile(
	optimizer="adam",
	loss="mean_squared_error",
	metrics=["mean_absolute_error"]
)
model.summary()

print("--- model fit ---")
history = model.fit(
	x_train,
	y_train,
	batch_size=1,
	epochs=5
)



print("--- all done ---")