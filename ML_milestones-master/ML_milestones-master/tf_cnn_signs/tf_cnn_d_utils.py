import h5py
import numpy as np
import matplotlib.pyplot as plt
import math

def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def preprocess_data(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, num_classes=6):

    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # Convert training and test labels to one hot matrices : 3 = [0, 0, 0, 1, 0, 0]
    Y_train = convert_to_one_hot(Y_train_orig, num_classes).T
    Y_test = convert_to_one_hot(Y_test_orig, num_classes).T

    return X_train, Y_train, X_test, Y_test

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:  X -- input data
                Y -- labels
                mini_batch_size -- size of the mini-batches, integer
                seed -- to keep the same results

    Returns:    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Partition (shuffled_X, shuffled_Y). - without counting the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def convert_to_one_hot(Y, C):
    """
    Convert to one hot vercors
    Arguments:  Y -- targets
                C -- number of classes
    Returns:    Y -- the one hot output
    """
    Y = np.eye(C)[Y.reshape(-1)].T # the .reshape(-1) is here to make sure we have the right labels format
    return Y

def show_image(index, X_orig, Y_orig):
    '''
    show the picture of index "index" and its label from the training dataset.
    in the train_set the index should be an integer < 1080
    '''
    if (index in range(1080)):
        plt.imshow(X_orig[index])
        print ("y = " + str(np.squeeze(Y_orig[:, index])))
        plt.show()
    else : print("the number of examples in this dataset is 1080")
