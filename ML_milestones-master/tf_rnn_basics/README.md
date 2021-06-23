# Recurrent Neural Networks - Basics

Recurrent neural networks (RNNs) are the class of artificial neural network that have "memory". They can read inputs one at a time, and remember some information/context through the hidden layer activations.

In this project, we present a basic implementation of a RNN with LSTM. The key idea in this latter is that the network can learn what to store in the long term state, what to throw away, and what to read from it.

I choose to make this project a 'one-file' one in order to show that the code of a basic RNN does not differ greatly from what we have seen earlier in the other TensorFlow application.

We will go deeper in RNNs in the next project, with a more complex case study.

We used the MNIST dataset and we have achieved a 0.998 training accuracy and 0.992 testing accuracy.
