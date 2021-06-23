import math
import scipy
import sklearn
from dnn import *
import numpy as np
from PIL import Image
from dnn_utils import *
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1)

# Genereting the data (two moons 2D)
X, y = make_moons(n_samples=1000, noise=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#---------------------------------------------------------------------------------------------------

def model(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9,
          beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True):
    """
    Deep neural network model which can be run in different optimizer modes.

    Arguments:  X -- input data - shape (1000, 2)
                Y -- labels (1 for blue dot / 0 for red dot) - shape (1000,)
                layers_dims -- python list, containing the size of each layer
                beta -- Momentum hyperparameter
                beta1 -- Exponential decay hyperparameter for the past gradients estimates
                beta2 -- Exponential decay hyperparameter for the past squared gradients estimates
                epsilon -- hyperparameter preventing division by zero in Adam updates
                num_epochs -- number of epochs
                print_cost -- True to print the cost every 1000 epochs

    Returns:    parameters -- python dictionary containing your updated parameters
    """

    L = len(layers_dims)             # number of layers in the neural networks
    costs = []                       # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update
    seed = 0                        # For grading purposes, so that your "random" minibatches are the same as ours

    # Initialize parameters
    parameters = initialize_parameters_deep(layers_dims)

    # Initialize the optimizer
    if optimizer == "gd":
        pass # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    # Optimization loop
    for i in range(num_epochs):

        # To reshuffle differently the dataset after each epoch
        seed = seed + 1
        # Define the random minibatches
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            a3, caches = L_model_forward(minibatch_X.T, parameters)

            # Compute cost
            cost = compute_cost(a3, minibatch_Y.T)

            # Backward propagation
            grads = L_model_backward(a3.T, minibatch_Y.T, caches)

            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1 # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2,  epsilon)

        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters

#Mini batch Gradient descent - Training
layers_dims = [X_train.shape[1], 5, 2, 1]
parameters = model(X_train, y_train, layers_dims, optimizer="gd")

#Mini batch Gradient descent - Prediction
predictions = predict(X_train, y_train, parameters, model="Gradient descent training")
predictions = predict(X_test, y_test, parameters, model="Gradient descent test")

# Plot decision boundary
plt.title("Model with Gradient Descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), X_train.T, y_train.T)

#Mini-batch gradient descent with momentum - Training
layers_dims = [X_train.shape[1], 5, 2, 1]
parameters = model(X_train, y_train, layers_dims, beta=0.9, optimizer="momentum")

#Mini batch Gradient descent with momentum - Prediction
predictions = predict(X_train, y_train, parameters, model="Momentum training")
predictions = predict(X_test, y_test, parameters, model="Momentum test")

# Plot decision boundary
plt.title("Model with Momentum optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), X_train.T, y_train.T)

#Mini-batch with Adam - Training
layers_dims = [X_train.shape[1], 5, 2, 1]
parameters = model(X_train, y_train, layers_dims, optimizer="adam", learning_rate=0.0001)

#Mini-batch with Adam - Prediction
predictions = predict(X_train, y_train, parameters, model="Adam training")
predictions = predict(X_test, y_test, parameters, model="Adam test")

# Plot decision boundary
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), X_train.T, y_train.T)
