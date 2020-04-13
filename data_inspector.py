import tensorflow as tf
import os

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'models')

model = tf.keras.models.load_model(filename)

print(model.summary())
