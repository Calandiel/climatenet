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

xdim = 180
ydim = 90
padding = 9
dd = 1 + padding * 2
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
koppens_weights = {
	0: 1., # water
	1: 1., # jungle
	2: 1., # monsoon
	3: 1., # savannah
	4: 1.,
	5: 1.,
	6: 1.,
	7: 1.,
	8: 1.,
	9: 1.,
	10: 1.,
	11: 1.,
	12: 1.,
	13: 1.,
	14: 1.,
	15: 1.,
	16: 1.,
	17: 1.,
	18: 1.,
	19: 1.,
	20: 1.,
	21: 1.,
	22: 1.,
	23: 1.,
	24: 1.,
	25: 1.,
	26: 1.,
	27: 1.,
}

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

	input_data = np.pad(input_data, ((padding, padding), (0, 0), (0, 0)), 'constant', constant_values=(0, 0))
	input_data=np.pad(input_data, ((0, 0), (padding, padding), (0, 0)), 'wrap')

	x_train.append(input_data)
	y_train.append(label_data)

	end_time = time.time()
	print(str(a) + ": " + str(end_time - start_time) + "s")

# Calculate weights
total = 28.0
for i in y_train[0]:
	for j in i:
		koppens_weights[np.argmax(j)] = koppens_weights[np.argmax(j)] + 1
		total = total + 1.0
for i in range(28):
	koppens_weights[i] = total / koppens_weights[i]

print("Image loaded!")

x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train[0].shape)
print(y_train[0].shape)

def get_sub_array(ni, xin, yin, slices_of_data):
	return slices_of_data[ni, yin:yin+2*padding+1, xin:xin+2*padding+1, :]
# For training
class DataGenerator(tf.keras.utils.Sequence):
	def __init__(self, batch_size, x_s, y_s, *args, **kwargs):
		self.batch_size = batch_size
		self.x_data = x_s
		self.y_data = y_s
		self.indices = np.array(range(xdim*ydim))
		np.random.shuffle(self.indices)

	def __len__(self):
		return int(np.floor(self.x_data.shape[0] * xdim *ydim / self.batch_size))

	def __getitem__(self, index):
		ni = int(index * self.batch_size) // (xdim * ydim)
		x = np.array([np.zeros((dd, dd, 6)) for o in range(self.batch_size)])
		y = np.array([np.zeros((len(koppens))) for o in range(self.batch_size)])

		for o in range(self.batch_size):
			ii = ni * self.batch_size + o
			ii = self.indices[ii]
			xin = ii % xdim
			yin = ii // xdim

			#ni = random.randint(0, self.x_data.shape[0] - 1) # index of the image from which we're copying data
			#xin = random.randint(0, xdim - 1)  # x of the pixel we're looking at, -1 is here because of inclusivity of randint
			#yin = random.randint(0, ydim - 1)  # y of the pixel we're looking at, -1 is here because of inclusivity of randint
			ooo = get_sub_array(ni, xin, yin, self.x_data)
			x[o] = ooo
			for i in range(len(koppens)):
				y[o, i] = self.y_data[ni, yin, xin, i]

		return x, y

	def on_epoch_end(self):
		np.random.shuffle(self.indices)
# For predicting
class DataProvider(tf.keras.utils.Sequence):
	def __init__(self, x_s, ni, batch_size, *args, **kwargs):
		self.x_data = x_s
		self.ni = ni
		self.batch_size = batch_size

	def __len__(self):
		return xdim * ydim

	def __getitem__(self, index):
		index_int = int(index)
		xin = index_int % xdim
		yin = index_int // xdim

		#print(f"{xin}, {yin}")
		x = np.array([np.zeros((dd, dd, 6)) for o in range(self.batch_size)])
		for o in range(self.batch_size):
			x[o] = get_sub_array(self.ni, xin, yin, self.x_data)
		return x

	def on_epoch_end(self):
		pass


model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(dd, dd, 6)))
#model.add(layers.Conv2D(8, kernel_size=(3, 3), activation='sigmoid'))
model.add(layers.Flatten())
#layers.Dropout(0.2)
model.add(layers.Dense(30, activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(30, activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(len(koppens), activation='softmax'))

print("--- compiling the model ---")
model.compile(
	optimizer='adam',#tf.keras.optimizers.SGD(learning_rate=0.0001),
		loss='categorical_crossentropy',
	metrics=["mean_squared_error", "categorical_accuracy"]
)
model.summary()

print("--- model fit ---")
gen = DataGenerator(int(xdim*ydim/8.0), x_train, y_train)
history = model.fit(
	gen,
	epochs=6400,
	workers=10,
	class_weight=koppens_weights
)

print("--- model predict ---")
# ID of the image in x_train that we want to export. 0 stands for Earth
image_id = 0
img_to_save = np.zeros((ydim, xdim, 3))
gen = DataProvider(x_train, image_id, 80)
results = model.predict(gen, workers=10, verbose=1)
ii = 0
for x in range(xdim):
	for y in range(ydim):
		#print(results[ii])
		img_to_save[y, x] = koppens[np.argmax(results[ii])] / 255.0
		ii = ii + 1
plt.imsave("export.png", img_to_save)

print("--- all done ---")
