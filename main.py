import os
import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import time
import random
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras import layers
from multiprocessing.dummy import Pool as ThreadPool



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

koppens = np.array([
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
])

x_train = []
y_train = []

for a in data:
	start_time = time.time()

	img_input = img_to_array(load_img("./training_data_inputs/" + a, color_mode='rgb'))
	img_label = img_to_array(load_img("./training_data_labels/" + a, color_mode='rgb'))

	input_data = np.zeros((img_input.shape[0], img_input.shape[1], 6))
	label_data = np.zeros((img_input.shape[0], img_input.shape[1], 28))

	for y in range(img_input.shape[0]):
		for x in range(img_input.shape[1]):
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
			for n in range(len(koppens)):
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
#print(x_train[0].shape)
#print(y_train[0].shape)

dd = 66 * 2 + 1

def get_sub_array(ni, xin, yin, slices_of_data):
	dim = int((dd - 1) / 2)
	low_x = xin - dim
	high_x = xin + dim
	low_y = yin - dim
	high_y = yin + dim

	top_right_x_low = 0
	top_right_x_high = 0
	top_left_x_low = 0
	top_left_x_high = 0
	if low_x < 0:
		top_left_x_low = low_x % slices_of_data.shape[2]
		top_left_x_high = slices_of_data.shape[2] + 1
		top_right_x_low = 0
		top_right_x_high = high_x + 1
	elif high_x >= slices_of_data.shape[2]:
		top_left_x_low = low_x
		top_left_x_high = slices_of_data.shape[2] + 1
		top_right_x_low = 0
		top_right_x_high = 1 + (high_x % slices_of_data.shape[2])
	else:
		top_right_x_low = low_x
		top_right_x_high = high_x + 1

	down_zeros_size = 0
	up_zeros_size = 0
	if low_y < 0:
		down_zeros_size = abs(low_y)
		low_y = 0
		high_y = high_y + 1
	elif high_y >= slices_of_data.shape[1]:
		up_zeros_size = 1 + high_y - slices_of_data.shape[1]
		high_y = slices_of_data.shape[1] + 1
	else:
		high_y = high_y + 1

	left_part = slices_of_data[ni, low_y:high_y, top_left_x_low:top_left_x_high, :]
	right_part = slices_of_data[ni, low_y:high_y, top_right_x_low:top_right_x_high, :]
	combined = np.concatenate((left_part, right_part), axis=1)
	combined = np.concatenate((
		np.zeros((down_zeros_size, dd, 6)),
		combined,
		np.zeros((up_zeros_size, dd, 6))
		), axis=0)
	return combined

class DataGenerator(tf.keras.utils.Sequence):
	def __init__(self, batch_size, x_s, y_s, *args, **kwargs):
		self.batch_size = batch_size
		self.x_data = x_s
		self.y_data = y_s

	def __len__(self):
		return 100 # int(np.floor(self.x_data.shape[0] / self.batch_size))

	def __getitem__(self, index):
		x = np.array([np.zeros((dd, dd, 6)) for o in range(self.batch_size)])
		y = np.array([np.zeros((len(koppens))) for o in range(self.batch_size)])

		for o in range(self.batch_size):
			ni = random.randint(0, self.x_data.shape[0] - 1) # index of the image from which we're copying data
			xin = random.randint(0, self.x_data.shape[2] - 1)  # x of the pixel we're looking at
			yin = random.randint(0, self.x_data.shape[1] - 1)  # y of the pixel we're looking at

			x[o] = get_sub_array(ni, xin, yin, self.x_data)
			for i in range(len(koppens)):
				y[o, i] = self.y_data[ni, yin, xin, i]

		return x, y

	def on_epoch_end(self):
		pass

model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(dd, dd, 6)))
model.add(layers.Conv2D(16, kernel_size=(2, 2), activation='relu'))
model.add(layers.Conv2D(8, kernel_size=(3, 3), strides=(2, 2), activation='sigmoid'))
model.add(layers.Conv2D(8, kernel_size=(3, 3), strides=(2, 2), activation='sigmoid'))
model.add(layers.Conv2D(8, kernel_size=(3, 3), strides=(2, 2), activation='sigmoid'))
model.add(layers.Flatten())
model.add(layers.Dense(len(koppens), activation='softmax'))

print("--- compiling the model ---")
model.compile(
	optimizer="adam",
	loss=tf.keras.losses.CategoricalCrossentropy(),
	metrics=["mean_absolute_error", "accuracy"]
)
model.summary()

print("--- model fit ---")
gen = DataGenerator(100, x_train, y_train)
history = model.fit(
	gen,
	epochs=1,
	workers=10
)

print("--- model predict ---")
start_time = time.time()
# ID of the image in x_train that we want to export. 0 stands for Earth
image_id = 0
img_to_save = np.zeros((x_train.shape[2], x_train.shape[1], 3))
pool = ThreadPool(10)
def map_func(ii):
	prediction_data = np.array([get_sub_array(0, x, ii, x_train)])
	cc = model.predict(prediction_data)
	return koppens[np.argmax(cc[0])] / 255.0
for x in range(x_train.shape[2]):
	print("X>"+str(x))
	results = np.array(pool.map(map_func, np.array(range(x_train.shape[1]))))
	img_to_save[x] = results
	end_time = time.time()
	print("time thus far: " + str(end_time - start_time) + ", ETA: " + str((x_train.shape[2] * (end_time - start_time) / (x + 1.0))))

#print("Image to save:")
#print(img_to_save)
#print(img_to_save.shape)
plt.imsave("export.png", img_to_save)
end_time = time.time()
print("Time to predict and save to file: " + str(end_time - start_time))

print("--- all done ---")
