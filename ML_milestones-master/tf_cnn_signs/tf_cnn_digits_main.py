import math
import h5py
import scipy
import numpy as np
from PIL import Image
import tensorflow as tf
from scipy import ndimage
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tf_cnn_d_fcts import model
from tf_cnn_d_utils import load_dataset, preprocess_data, show_image

np.random.seed(1)

# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

#show an image from the trainnig dataset
#show_image(1000, X_train_orig, Y_train_orig)

# Cleaning the dataset
X_train, Y_train, X_test, Y_test = preprocess_data(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig)

# Training the parameters
parameters = model(X_train, Y_train, X_test, Y_test)

#Train Accuracy: 0.8898148
#Test Accuracy: 0.7916667
