# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 00:09:07 2020

@author: Hafiz
"""
from imutils import paths
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
from keras.applications import ResNet50
import pickle
#from io_.hdf5datasetgenerator import HDF5DatasetGenerator
#import config.fruits_ml_web_app as config
import csv
from sklearn import preprocessing
import os

bs = 64

#submissionGen = HDF5DatasetGenerator(config.TEST_PRED_HDF5, bs, binarize = False) # providing the path to training hdf file
model = ResNet50(weights='imagenet', include_top = False)

#sb = submissionGen.generator(images_only = True)
#features = model.predict_generator(submissionGen.generator(images_only = True), steps= np.int(np.ceil(submissionGen.numImages / bs)), callbacks=None, max_queue_size= 2 * bs, workers=1, use_multiprocessing=False, verbose=1)
#print(features.shape)

imagePath = r"/Test/Apple Braeburn/3_100.jpg" # Enter image direcotry
image = load_img(imagePath, target_size = (224, 224))
image = img_to_array(image)

# preprocess the image by (1) expanding the dimension and 
# (2) subtracting the mean RGB pixel intensity from the 
# ImageNet dataset
image = np.expand_dims(image, axis = 0)
image = imagenet_utils.preprocess_input(image)

features = model.predict(image)
features = features.reshape((features.shape[0], 7 * 7 * 2048))
model_linear_regression = pickle.load(open(r"outputs/fruits_ml_web_app_C_1.model", "rb"))
print(features.shape)
predictions = model_linear_regression.predict(features)
le = preprocessing.LabelEncoder()
# getting back the labels (saved by extract_features.py) from the path to pass to fit labelEncoder such that knowing these actual labels it can decode the encoded label to these actual labels
with open(r"outputs/labels.txt", "rb") as fp:   # Unpickling (it was saved in binary format)
    labels = pickle.load(fp)

le.fit(labels) # fitting the labels such that le can be used to go back to the actual labels from the predictions
actual_pred_labels = le.inverse_transform(predictions)

print(actual_pred_labels[0])
