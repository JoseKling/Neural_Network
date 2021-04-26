## Neural_Network
This is an implementation of a Neural Network in Python using only numpy.  
There are 2 classes: NeuralNet and Layer. 

NeuralNet is where the whole model will be stored. It must be initialized with the number of inputs and outputs of the model and the desired cost function (for now there is only one). Its methods:
* add_layer - Adds a hidden layer or the output layer. 
* train - Train the model using a dataset.
* backpropagation - Makes a forward pass and propagates back the gradient to update the weights during training.
* forward_pass - Makes a forward pass with the given weights and inputs.
* average_error - Calculates the average error of the model for a given dataset.
* get_error - Calculates the error for a given data point. The error function is chosen when initializing the NeuralNet class.
* predict - Predicts the classes of a set of datapoints
* score - Gets the percentage of correct predictions for a dataset
* squared_loss - The cost function implemented.

The class Layer creates a layer with desired number of nodes and activation function (ReLU, Leaky ReLU, tanh, sigmoid or softmax). Methods:
* forwards - A forward pass through this layer with given inputs. Must also store values needed for the backpropagation.
* backwards - Calculates the gradients for the nodes and the weights given some input gradients from the previous layer.
* relu
* leaky_relu
* sigmoid
* tanh
* softmax
