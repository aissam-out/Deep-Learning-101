# Understanding CNNs -TensorFlow Digits dataset-

In this project brings you with the fundamentals to understand how Convolutional Neural Networks work, first by presenting the building bloc functions to implement convolutional and pooling layers from scratch (just with numpy), and then by exposing a stadard and basic fully functioning ConvNet using TensorFlow.

## Data

As a dataset we use a subset of the SIGNS dataset

Training set: 1080 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (180 pictures per number).
Test set: 120 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (20 pictures per number)



## Model

- CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

- Mini-batch gradient descent with minibatch_size of 64

