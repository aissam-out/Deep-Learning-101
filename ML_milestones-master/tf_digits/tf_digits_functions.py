import tensorflow as tf
from tensorflow.python.framework import ops
from tf_digits_utils import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy
import numpy as np
from PIL import Image
from scipy import ndimage

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session

    Arguments:  n_x -- scalar, size of an image vector (height * width * depth)
                n_y -- scalar, number of classes
    Returns:    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
                Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    """

    X = tf.placeholder(tf.float32, shape=[n_x, None])
    Y = tf.placeholder(tf.float32, shape=[n_y, None])

    return X, Y


def initialize_parameters(layer_dims):
    """
    Initializes parameters to build a neural network with tensorflow.
    Xavier Glorot and Yoshua Bengio (2010)

    Returns:    parameters -- a dictionary of tensors containing weights and biases
    """

    tf.set_random_seed(1)
    L = len(layer_dims) - 1
    parameters = {}

    for l in range(L):
        parameters["W"+str(l+1)] = tf.get_variable("W"+str(l+1),
                                                   [layer_dims[l+1], layer_dims[l]],
                                                   initializer=tf.contrib.layers.xavier_initializer(seed = 1))

        parameters["b"+str(l+1)] = tf.get_variable("b"+str(l+1),
                                                   [layer_dims[l+1], 1],
                                                   initializer=tf.zeros_initializer())

    return parameters


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: (L-1)[LINEAR->RELU]->LINEAR->SOFTMAX

    Arguments:  X -- input dataset placeholder, of shape (input size, number of examples)
                parameters -- python dictionary containing model parameters
    Returns:    ZL -- the output of the last LINEAR unit
    """

    L = len(parameters) // 2
    A = X
    Z = None

    for l in range(L):
        W = parameters["W"+str(l+1)]
        b = parameters["b"+str(l+1)]

        Z = tf.add(tf.matmul(W, A), b)
        A = tf.nn.relu(Z)

    return Z


def compute_cost(ZL, Y):
    """
    Computes the cost

    Arguments:  ZL -- output of the last LINEAR unit of forward propagation. shape (number of classes, number of examples)
                Y -- "true" labels vector placeholder, same shape as ZL
    Returns:    cost -- Tensor of the cost function
    """

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(ZL)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=32, print_cost=True):
    """
    Implements a L-layer tensorflow neural network: (L-1)[LINEAR->RELU]->LINEAR->SOFTMAX

    Arguments:  X_train -- training set, of shape (input size, number of training examples)
                Y_train -- test set, of shape (output size, number of training examples)
                X_test -- training set, of shape (input size, number of training examples)
                Y_test -- test set, of shape (output size, number of test examples)
                print_cost -- True to print the cost every 100 epochs
    Returns:    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()           # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape            # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]              # n_y : output size
    costs = []                          # To keep track of the cost
    layer_dims = [n_x, 25, 12, n_y]     # to initialize parameters [12288, 25, 12, 6]

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters(layer_dims)

    # Forward propagation: Build the forward propagation in the tensorflow graph
    ZL = forward_propagation(X, parameters=parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(ZL, Y)

    # Adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                            # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot and save the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.savefig('cost.png')

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters


def predict(X, parameters, n_x):

    L = len(parameters) // 2
    params = {}

    for l in range(L):
        params["W"+str(l+1)] = tf.convert_to_tensor(parameters["W"+str(l+1)])
        params["b"+str(l+1)] = tf.convert_to_tensor(parameters["b"+str(l+1)])

    x = tf.placeholder("float", [n_x, 1])

    zl = forward_propagation_for_predict(x, params)
    last = tf.argmax(zl)

    sess = tf.Session()
    prediction = sess.run(last, feed_dict={x: X})

    return prediction


def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: (L-1)[LINEAR -> RELU] -> LINEAR -> SOFTMAX

    Arguments:  X -- input dataset placeholder, of shape (input size, number of examples)
                parameters -- python dictionary containing parameters
    Returns:    ZL -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    L = len(parameters) // 2
    A = X
    Z = None

    for l in range(L):
        W = parameters["W"+str(l+1)]
        b = parameters["b"+str(l+1)]

        Z = tf.add(tf.matmul(W, A), b)
        A = tf.nn.relu(Z)

    return Z


def test_own_image(own_image, parameters):
    """
    test the algorithm with your own image
    """
    # Preprocessing of the image to fit the algorithm.
    fname = "images/" + own_image
    image = np.array(ndimage.imread(fname, flatten=False))
    own_image = scipy.misc.imresize(image, size=(64, 64)).reshape((1, 64 * 64 * 3)).T
    own_image_prediction = predict(own_image, parameters, 64 * 64 * 3)

    plt.imshow(image)
    print("Your algorithm predicts: y = " + str(np.squeeze(own_image_prediction)))
