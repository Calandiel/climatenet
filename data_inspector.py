import tensorflow as tf

model = tf.keras.models.load_model("models/saved_model.pb")

print(model.summary())