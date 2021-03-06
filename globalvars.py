####################
### MODEL PARAMS ###
####################
RADIUS = 2 # 4 # how many cells in any direction do we use?


#######################
### TRAINING PARAMS ###
#######################
EPOCHS = 10
EPOCH_LENGHT_MULTIPLIER = 1
BATCH_SIZE = 32
VALIDATION_LENGTH_MULTIPLIER = 0.001


################################
### MISC AND LOGGING CONTROL ###
################################
VERBOSITY = 1 #Verbosity of training. Options: 0 - silent, 1 - progress bar, 2 - no progress bar, update every epoch
JUST_ONE_FILE = False # IF TRUE, IT WILL ONLY LOAD A SINGLE GRIB FILE
PRINT_UNUSED = False # IF TRUE, THE GAME WILL PRINT BAND DATA, THEIR UNITS, NAMES, SYMBOLS AND LAYERS
PRINT_USED = False
SHOULD_PLOT = True # IF TRUE, IT WILL GENERATE A PLOT FOR EVERY RASTER BAND
SHOULD_DISPLAY_BAND_STATS = True # IF TRUE, DISPLAYS MAX/MIN OF PRINTED RASTER BANDS
SHOULD_SAVE_MODEL = True


####################
### DECLARATIONS ###
####################
GLOBAL_MAP_DIMENSIONS = (-1, -1)
INPUT_SIZE = -1
OUTPUT_SIZE = -1