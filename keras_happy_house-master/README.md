# keras_happy_house

Keras offers consistent & simple APIs in order to enable deep learning engineers to build their models very quickly. Just as TensorFlow is a higher-level framework than Python, Keras is an even higher-level framework and provides additional abstractions. It reduces the number of code lines and provides clear feedback upon user error.

Therefore, unless the project you are working on has some research purpose or is built on special kind of neural network, you'd better rely on Keras; it’s pretty simple to quickly build even very complex models in Keras.

In this project, we are implementing the "Happy House" problem, where we allow a person to enter the house only if he/she is smiling! So, a smile detector! This problem was presented in the Deep Learning Specialization CNN - coursera.

## Problem statement

To handle this problem of evaluating the current state of happiness of your guests, and given the fact that you are a deep learning expert, you are going to build an algorithm which uses pictures from the front door camera to check if the person is happy or not. The door should open only if the person is happy.

## Data

Data is composed of pictures of your friends and yourself, taken by the front-door camera. The dataset is labbeled: 1-"happy" 0-"not happy"

The train set has 600 examples. The test set has 150 examples.

Source: This dataset is gathered from the Deep Learning Specialization - Convolutional Neural Network Course - Week 2 - Happy House Exercise.

## Model

 ```input_1 : InputLayer``` --> ```zero_padding2d_1 : ZeroPadding2D``` --> ```conv0 : Conv2D```
 
 --> ```bn0 : BatchNormalization``` -->
 
 ```activation_1 : Activation``` --> ```max_pool : MaxPooling2D``` --> ```flatten_1 : Flatten```
 
 --> ```fc : Dense```

## Evaluation

Number of epochs = 40

Train accuracy = 0.9983

Test accuracy = 0.9733
