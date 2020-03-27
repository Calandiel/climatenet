import os
import pathlib
import matplotlib as plt
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
    return size * size * len(nparrays) + 2 # + 2 at the end for "y" and "time
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
            plt.pyplot.imshow(data, cmap='jet')
            plt.pyplot.title(ttl)
            plt.pyplot.show()

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

model = get_model()

# NOTE !!! EVEN THO BELOW WE USE A WORD "DAY" WE REALLY MEAN "TICK"
if len(data_by_days) >= 2:
    generator = DataGenerator(data_by_days)
    print("Generator len: " + str(len(generator)))
    model.fit(generator, callbacks=[g.CP_CALLBACK])
else:
    print("NOT ENOUGH GRIB FILES FOR ACTUAL LEARNING!")
