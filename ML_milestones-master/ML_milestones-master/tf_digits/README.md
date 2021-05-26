# TensorFlow -Digits dataset-

Until now, we've always used numpy to build NNs.
Here we are using a DL framework that will allow us to build NNs more easily: TensorFlow

## Data

As a dataset we use a subset of the SIGNS dataset

*Training set*: 1080 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (180 pictures per number).

*Test set*: 120 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (20 pictures per number)

## The model

The Neural Network we are implementing is of shape [64 \* 64 \* 3=12288, 25, 12, 6]

We use Mini-batch gradient descent with minibatch_size of 32

The model is LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX.

## To keep in mind

To code in TensorFlow you have to respect the following steps:

- Create a graph containing Tensors (Variables, Placeholders ...) and Operations (tf.matmul, tf.add, ...)

- Create a session

- Initialize the session

- Run the session to execute the graph

