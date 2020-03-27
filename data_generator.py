import tensorflow as tf
import numpy as np
from random import randrange

import globalvars as g

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, data, batch_size = 32):
        self.data, self.batch_size = data, batch_size
    def __len__(self):
        total = g.GLOBAL_MAP_DIMENSIONS[0] * g.GLOBAL_MAP_DIMENSIONS[1]
        return int(total / float(self.batch_size))

    #returns a batch
    def __getitem__(self, index):        
        X = np.empty((self.batch_size, g.INPUT_SIZE))
        Y = np.empty((self.batch_size, g.OUTPUT_SIZE))
        for i in range(self.batch_size):
            X[i,], Y[i,] = self.get_pair(index)
        #xnan = np.isnan(np.sum(X))
        #print("X NaN: " + str(xnan))
        #ynan = np.isnan(np.sum(Y))
        #print("Y NaN: " + str(ynan))
        #if xnan or ynan:
        #    raise SystemError("NANS!")
        return X, Y
    def get_pair(self, index):
        # construct data
        i = randrange(len(self.data) - 1)
        x, y = randomrange = randrange(g.GLOBAL_MAP_DIMENSIONS[1]), randrange(g.GLOBAL_MAP_DIMENSIONS[0])

        current_tick = self.data[i]
        next_tick = self.data[i + 1]
        current_tick_data = np.zeros(g.INPUT_SIZE)
        next_tick_data = np.zeros(g.OUTPUT_SIZE)

        construct_input(current_tick, current_tick_data,
                             x, y,
                             0, i)
        construct_output(next_tick, next_tick_data, x, y)
        return current_tick_data, next_tick_data
    
def construct_input(nparrays, data_array,
                    x, y, 
                    day, time):
    #print(len(nparrays))
    i = 0
    for ii in range(len(nparrays)):
        layer = nparrays[ii]
        for xx in range(g.RADIUS * 2 + 1):
            xx = (xx - g.RADIUS + x) % g.GLOBAL_MAP_DIMENSIONS[1]
            for yy in range(g.RADIUS * 2 + 1):
                yy = max(0, min((yy - g.RADIUS + y), g.GLOBAL_MAP_DIMENSIONS[0] - 1))
            
                data_array[i] = layer[yy, xx]
                i += 1
def construct_output(nparrays, data_array, x, y):
    dim_x, dim_y = nparrays[0].shape
    for i in range(len(nparrays)):
        layer = nparrays[i]
        data_array[i] = layer[y, x]






