import tensorflow as tf

model = tf.keras.models.load_model("models/latest.tf")

print(model.summary())
