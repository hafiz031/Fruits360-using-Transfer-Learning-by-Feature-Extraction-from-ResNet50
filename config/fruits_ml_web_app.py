'''
A configuration file for path management
'''

# Project Base
PROJECT_BASE = '' # please change it according to where the whole project resides

# define the path to the images directory
IMAGES_PATH = PROJECT_BASE + '/Train'
# since we do not have access to validation data we need to take a number of images from train and test on them
NUM_CLASSES = 120
BATCH_SIZE = 16
BUFFER_SIZE = 1000
N_JOBS = -1

# define the file path to output training, validation and testing HDF5 files
FEATURES_HDF5 = PROJECT_BASE + '/hdf5/features.hdf5'
VAL_HDF5 = PROJECT_BASE + '/hdf5/val.hdf5'
TEST_HDF5 = PROJECT_BASE + '/hdf5/test.hdf5'

# path to the output model file
MODEL_PATH = PROJECT_BASE + '/outputs/resnet50_transfer_learning_fruits_ml_web.model'

# define the path to the output directory used for storing plots, classification_reports etc.
OUTPUT_PATH = PROJECT_BASE + '/outputs'
