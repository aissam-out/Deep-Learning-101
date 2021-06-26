import scipy
import numpy as np
from PIL import Image
from scipy import ndimage
from tf_digits_functions import model, predict, test_own_image
from tf_digits_utils import load_dataset, preprocess_data

np.random.seed(1)

# Parameters
num_classes = 6

# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Cleaning the dataset
X_train, Y_train, X_test, Y_test = preprocess_data(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, num_classes)

# Training the parameters
parameters = model(X_train, Y_train, X_test, Y_test)

# Predict using a sample image of test dataset
predicted_class = predict(X_test[:,1:2], parameters, X_test.shape[0])

# Test with your own image
test_image = "good_job.jpg"
test_own_image(test_image, parameters)
