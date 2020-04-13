import tensorflow as tf
import globalvars as g

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

    model.compile(optimizer='adadelta',
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error', 
                           'mean_squared_logarithmic_error',
                           'mean_squared_error',
                           'logcosh'])
    return model