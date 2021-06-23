# TensorFlow Classification: MNIST digits with CNN and Regression

Build and train a convolutional neural-network (CNN) and a Regression model for classifying MNIST digits dataset


## Network architecture

CNN with 4 layers has following architecture.

- input layer : 784 nodes (MNIST images size)
- first convolution layer : 5x5x32
- first max-pooling 2\*2 layer
- second convolution layer : 5x5x64
- second max-pooling 2\*2 layer
- fully-connected layer : 1024 nodes
- dropout
- output layer : softmax 10 nodes (number of class for MNIST)



