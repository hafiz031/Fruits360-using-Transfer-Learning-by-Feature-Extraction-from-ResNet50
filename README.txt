# Fruits 360

The project is basically the solution to the fruits detection and the dataset is collected from: https://www.kaggle.com/moltean/fruits

# Solution approach:

-> It is a transfer learning based approach of solving this classification problem
-> A pretrained ResNet50 model (pretrained on ImageNet Dataset) is taken as a feature extractor 
-> Using the feature vector a logistic regression classifier is trained on it for making the final detection

# Training

-> The first step is to download the dataset, so after downloading the dataset extract them inside the corresponding 	Train and Test folder. The folders are kept empty right now (only having the snapshots how the folder will look   like after putting the dataset inside of them. You can downlaod the dataset from: https://www.kaggle.com/moltean/Fruits


-> Analysing the dataset I found that the images of each of the classes are just taken by rotating the same object. 	Probably the dataset was built by extracting the frames from a video. Hence, the consecutive frames were almost 	similar. So, I reduced the dataset by keeping consecutive images after a certain interval and deleting the rest.
	mini_dataset_generator.py does this. Modify the keep frequency in the code. DON'T FORGET TO BACKUP THE ORIGINAL DATASET AS IT WILL DELETE IMAGES MAINTAINING CERTAIN INTERVAL. Change the directory as required.

-> After that images are fed into ResNet50 model to get the feature vector out of it. For this reason the model's top was removed beforehand. Collecting these features the features were saved in a HDF5 file. As HDF5 is a binary format it has less IO overhead and training becomes faster. The extract_features.py implements this section.

-> After we get our train.hdf file we are good to go for transfer learning. 'train_model.py' implements this. For 	   this purpose I have used transfer learning using feature extraction by passing the extracted features to train a Logistoc Regression Classifier.

-> I have used GridSearchCV to find the best combination of parameters. The model worked fine for C = 1 (fruits_ml_web_app_C_1.model). The paramerer C is just the inverse of regularization parameter.

-> There are two file named predict.py and predictor.py and both are capable of making the final prediction. I wrote the predict.py file just to test with a single image. Fill up the directory of the image and it will print the fruit class in console. On the other hand predictor.py just 

-> The model was developped in standalone keras. If you are using Tensorflow-keras then change all 'keras' to tensorflow.keras


# Project structure:

├── config
│   ├── fruits_ml_web_app.py
│   └── __pycache__
│       ├── dogs_vs_cats_config.cpython-36.pyc
│       ├── dogs_vs_cats_ResNet_t_learning_config.cpython-36.pyc
│       ├── fruits_ml_web_app.cpython-36.pyc
│       └── googlenet_cifar10_config.cpython-36.pyc
├── extract_features.py
├── hdf5
├── mini_dataset_generator.py
├── outputs
│   ├── best_parameters.txt
│   ├── fruits_ml_web_app_C_0.005.model
│   ├── fruits_ml_web_app_C_1.model
│   ├── labels.txt
│   └── README.txt
├── predict.py
├── README.txt
├── Test
│   └── sample_test.png
├── Training
│   └── sample_train.png
└── train_model.py

6 directories, 17 files
