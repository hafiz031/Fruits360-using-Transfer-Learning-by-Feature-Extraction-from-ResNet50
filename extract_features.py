# import the necessery packages
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from io_.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
#import argparse
import random
import os
import config.fruits_ml_web_app as config
import pickle # for saving txt

#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required = True,
#                help = "path to input dataset")
#ap.add_argument("-o", "--output", required = True,
#                help = "path to HDF5 file")
#ap.add_argument("-b", "--batch-size", type = int, default = 16,
#                help = "batch size of images to be passed through network")
#ap.add_argument("-s", "--buffer-size", type = int, default = 1000,
#                help = "size of feature extraction buffer")
#args = vars(ap.parse_args())

# store the batch size in a convenience variable
#bs = args["batch-size"]
bs = config.BATCH_SIZE

# grab the list of images that we'll be describing than randomly 
# shuffle them to allow for easy training and testing splits via
# array slicing during training time
print("[INFO] loading images...")
#imagePaths = list(paths.list_images(args["dataset"]))
imagePaths = list(paths.list_images(config.IMAGES_PATH))
random.shuffle(imagePaths)
#print(imagePaths)
# extract the class labels from the image paths than encode the labels
labels = [p.split(os.path.sep)[-2] for p in imagePaths]

# saving the labels such that these labels can be used while we need to convert the encoded labels to actual labels back
with open(r"./outputs/labels.txt", "wb") as fp:   #Pickling
    pickle.dump(labels, fp) # this will save the lhe labels

le = LabelEncoder()
labels = le.fit_transform(labels)

# load the ResNet50 network
print("[INFO] loading network...")
model = ResNet50(weights = "imagenet", include_top = False) # pretrained ResNet architecture

# initialize the HDF5 dataset writer, than store the class label
# names in the dataset
#dataset = HDF5DatasetWriter((len(imagePaths), 7 * 7 * 2048),
#                            args["output"], dataKey = "features", bufSize = args["buffer_size"])
dataset = HDF5DatasetWriter((len(imagePaths),  7 * 7 * 2048),
                            config.FEATURES_HDF5 , dataKey = "features", data_dtype = "float", bufSize = config.BUFFER_SIZE)
dataset.storeClassLabels(le.classes_)

# initialize the progress bar
widgets = ["Extracting Features: ", progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval = len(imagePaths), 
                               widgets = widgets).start()

# loop over the image in batches
for i in np.arange(0, len(imagePaths), bs):
    # extract the batch of image and labels, than initialize the 
    # list of actual image that will be passed through the network
    # for feature extraction
    batchPaths = imagePaths[i : i + bs]
    batchLabels = labels[i : i + bs]
    batchImages = []
    
    # loop over the images and labels in the current batch
    for (j, imagePath) in enumerate(batchPaths):
        # load the input image using the Keras helper utility
        # write ensuring the image is resized to 224x224 pixels
        image = load_img(imagePath, target_size = (224, 224))
        image = img_to_array(image)
        
        # preprocess the image by (1) expanding the dimension and 
        # (2) subtracting the mean RGB pixel intensity from the 
        # ImageNet dataset
        image = np.expand_dims(image, axis = 0)
        image = imagenet_utils.preprocess_input(image)
        
        # add the image to the batch
        batchImages.append(image)
        
    # pass the images through the network and use the output as 
    # our actual features
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size = bs)
    
    # reshape the features so that each image is represented by
    # a flattened feature vector of the "MaxPooling" outputs
    features = features.reshape((features.shape[0], 7 * 7 * 2048)) # the output from the ResNet (top removed) for each batch is (batch_size, 7 * 7 * 2048) [2048s 7x7 activations]

    # add the features and labels to out HDF5 dataset
    dataset.add(features, batchLabels)
    pbar.update(i)
    
# close the dataset
dataset.close()    
pbar.finish()