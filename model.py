import tensorflow as tf
import globalvars as g
import os
import pathlib

def get_model():
    # Model goes here
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(g.INPUT_SIZE, activation='sigmoid', input_dim=g.INPUT_SIZE),
        tf.keras.layers.Dense(g.INPUT_SIZE * 2, activation='sigmoid'),
        tf.keras.layers.Dense(g.INPUT_SIZE * 2, activation='sigmoid'),
        tf.keras.layers.Dense(g.INPUT_SIZE * 2, activation='sigmoid'),
        tf.keras.layers.Dense(g.OUTPUT_SIZE, activation='sigmoid')
    ])
    checkpoint_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "checkpoints")
    #checkpoint_path = os.path.join(checkpoint_path, 'climate.nn')
    g.CP_CALLBACK = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 verbose=10,
                                                 save_freq=50)
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error', 
                           'mean_absolute_percentage_error',
                           'mean_squared_error'])
    return model