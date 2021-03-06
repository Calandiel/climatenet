import os
import pathlib
import matplotlib.pyplot as plt
import gdal
import tensorflow as tf
import numpy as np
import sys

import globalvars as g

from data_generator import DataGenerator
from model import get_model
from included_vars import data_vars, vars_to_plot, operators

print('Python version: %s' % sys.version)
print('TensorFlow version: %s' % tf.__version__)
print('Keras version: %s' % tf.keras.__version__)
    
def get_band_identifier(band_data):
    desc = band_data.GetDescription()
    metadata = band_data.GetMetadata()     
    d = str(metadata['GRIB_ELEMENT']) + " -- "
    d += str(metadata['GRIB_COMMENT']) + " -- "
    d += str(desc)
    return d
def print_band_identifier(ttl, data = None, used = True):
    if used and g.PRINT_USED:
        print(ttl)
        if g.SHOULD_DISPLAY_BAND_STATS:
                print("MAX: " + str(np.max(data)))
                print("MIN: " + str(np.min(data)))
    elif not used and g.PRINT_UNUSED:
        print(ttl + " # UNUSED")
        if g.SHOULD_DISPLAY_BAND_STATS:
                print("MAX: " + str(np.max(data)))
                print("MIN: " + str(np.min(data)))
    
def get_input_dimensions(nparrays):
    size = g.RADIUS * 2 + 1
    return size * size * len(nparrays) + 4 # x, y, day, time
def get_output_dimensions(nparrays):
    return len(nparrays)

# Loop over all data files
path = os.path.join(pathlib.Path(__file__).parent.absolute(), "data")
finished_files_count = 0
data_by_days = []
for ff in os.listdir(path):
    if g.JUST_ONE_FILE and finished_files_count > 0: break;
    print("Opening: " + ff)
    file_path = os.path.join(path, ff)

    # Open the file
    grib = gdal.Open(file_path)

    present_data_vars = []

    # Loop over all data fields of a grib file and load those requested to present_data_fields
    for a in range(grib.RasterCount):
        a += 1
        # Read an specific band
        band = grib.GetRasterBand(a)
        ttl = get_band_identifier(band)
        # Read the band as a Python array
        data = band.ReadAsArray()
        if ttl in data_vars:
            print_band_identifier(ttl, data = data, used = True)
        else:
            print_band_identifier(ttl, data = data, used = False)
        
        # Show the image
        if g.SHOULD_PLOT and ttl in vars_to_plot:
            plt.imshow(data, cmap='jet')
            plt.title(ttl)
            plt.show()

        # Add data from this layer to data fields
        if ttl in data_vars:
            # transform data
            if ttl in operators:
                op = operators[ttl]
                data *= op[1]
                data += op[0]
            else:
                raise SystemError("MISSING OPERATOR FOR: " + str(ttl))
            present_data_vars.append((ttl, data))
    
    # Verify that all requested data fields are present and that we don't have any excess fields either
    requested_data_vars = data_vars.copy()
    for a in present_data_vars:
        ttl = a[0]
        data = a[1]
        if ttl in requested_data_vars:
            requested_data_vars.remove(ttl)
        else:
            raise SystemError("PRESENT_DATA_VARS HAS AN ENTRY THAT WASN'T REQUESTED OR THERE IS A DUPLICATE!")
    if len(requested_data_vars) > 0:
        raise SystemError("NOT ALL REQUESTED FIELDS WERE PRESENT! MISSING: " + str(requested_data_vars))
    # Sort present_data_vars by ttl and the order in data_vars
    grib_data = []
    for i in range(len(data_vars)):
        ttl = data_vars[i]
        for a in present_data_vars:
            if a[0] == ttl:
                grib_data.append(a[1])
                continue
    data_by_days.append(grib_data)
    finished_files_count += 1
    print("--- complete ---")

g.GLOBAL_MAP_DIMENSIONS = data_by_days[0][0].shape
print("GLOBAL MAP DIMENSIONS: " + str(g.GLOBAL_MAP_DIMENSIONS))
g.INPUT_SIZE = get_input_dimensions(data_by_days[0])
print("INPUT SIZE: " + str(g.INPUT_SIZE))
g.OUTPUT_SIZE = get_output_dimensions(data_by_days[0])
print("OUTPUT SIZE: " + str(g.OUTPUT_SIZE))

mirrored_strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))
with mirrored_strategy.scope():
	model = get_model()

# NOTE !!! EVEN THO BELOW WE USE A WORD "DAY" WE REALLY MEAN "TICK"
if len(data_by_days) >= 2:
    generator = DataGenerator(
        data_by_days, 
        batch_size=g.BATCH_SIZE*mirrored_strategy.num_replicas_in_sync, 
        len_multiplier=g.EPOCH_LENGHT_MULTIPLIER)
    validation_generator = DataGenerator(
        data_by_days, 
        batch_size=g.BATCH_SIZE*mirrored_strategy.num_replicas_in_sync,
        len_multiplier=g.VALIDATION_LENGTH_MULTIPLIER)
    print("Generator len: " + str(len(generator)))
    print(model.summary())
    
    epochs_count = g.EPOCHS
    history = model.fit(generator,
              epochs=epochs_count,
              verbose=g.VERBOSITY,
              validation_data=validation_generator)
    
    # LOG STUFF
    print(history.history.keys())
    #  "Accuracy"
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('mean_absolute_error')
    plt.ylabel('mean_absolute_error')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


    # SAVE MODEL
    if g.SHOULD_SAVE_MODEL:
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'models')
        tf.keras.models.save_model(model, filename, overwrite=True)
else:
    print("NOT ENOUGH GRIB FILES FOR ACTUAL LEARNING!")


