RADIUS = 2 # 4 # how many cells in any direction do we use?
VERBOSITY = 2 #Verbosity of training. Options: 0 - silent, 1 - progress bar, 2 - no progress bar, update every epoch
# IF TRUE, IT WILL ONLY LOAD A SINGLE GRIB FILE
JUST_ONE_FILE = False
# IF TRUE, THE GAME WILL PRINT BAND DATA, THEIR UNITS, NAMES, SYMBOLS AND LAYERS
PRINT_UNUSED = False
PRINT_USED = False
# IF TRUE, IT WILL GENERATE A PLOT FOR EVERY RASTER BAND
SHOULD_PLOT = True
# IF TRUE, DISPLAYS MAX/MIN OF PRINTED RASTER BANDS
SHOULD_DISPLAY_BAND_STATS = True

GLOBAL_MAP_DIMENSIONS = (-1, -1)
INPUT_SIZE = -1
OUTPUT_SIZE = -1


# USED TO SAVE THE MODEL
CP_CALLBACK = None
SHOULD_SAVE_MODEL = True

# FOR THE SUPERCOMPUTER
MULTIPLE_GPUS = False
GPU_COUNT = 2