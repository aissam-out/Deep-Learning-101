# Build and apply a deep neural network to supervised learning : image classification with the two moons dataset


This project provides you with a deep neural network architecture designed from scratch without the use of any powerful framework. It is built over scikit learn.

The aim of exposing this code is to understand how NNs are built, their building blocs and how they process data; in other words, you will understand how machine learning algorithms think

As a dataset, we use the two moons 2D one, imported from the sklearn's datasets

The project is decomposed into three files:

- dnn_utils : contains some basic functions like relu, sigmoid and their backwards

- dnn	    : implements the most important functions needed to make a Deep Neural Network, namely the initialization process, forward propagation, compute cost, backward propagation as well as prediction and plot functions

- imgClassNn: the main file. It contains the core function "model()" which agregate all the subfunctions needed to create a neural net. This file is the one responsible of lunching the training process, printing and evaluating the results

The DeepNN implemented in this model can be run using different optimizer modes. In the implementation presented here we exposed it with gradient descent (gd) optimizer, gd with momentum optimizer and with Adam optimizer.
