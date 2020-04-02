import tensorflow as tf
import globalvars as g
import os
import pathlib

def get_model():
    # Model goes here
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(g.INPUT_SIZE, activation='relu', input_dim=g.INPUT_SIZE),
        tf.keras.layers.Dense(g.INPUT_SIZE * 2, activation='relu'),
        tf.keras.layers.Dropout(rate=0.15),
        tf.keras.layers.Dense(g.INPUT_SIZE * 2, activation='relu'),
        tf.keras.layers.Dropout(rate=0.15),
        tf.keras.layers.Dense(g.INPUT_SIZE * 2, activation='relu'),
        tf.keras.layers.Dropout(rate=0.15),
        tf.keras.layers.Dense(g.OUTPUT_SIZE, activation='relu')
    ])
    checkpoint_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "checkpoints")
    #checkpoint_path = os.path.join(checkpoint_path, 'climate.nn')
    if g.SHOULD_SAVE_MODEL:
        g.CP_CALLBACK = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                           verbose=1,
                                                           save_freq=1000)
    model.compile(optimizer='adadelta',
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error', 
                           'mean_squared_logarithmic_error',
                           'mean_squared_error',
                           'logcosh'])
    return model