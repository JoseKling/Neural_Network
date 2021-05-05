## Neural_Network
This is an implementation of a Neural Network in Python using only numpy.  
The classes are the following:

#NeuralNet 
Where the whole model will be stored. It must be initialized with the number of inputs and outputs of the model and the desired cost function (for now there is only one). Its methods:
* add_FC_layer - Adds a Fully Connected layer to the hidden layers.
* add_Conv_layer - Adds a Convolutional layer to the hidden layers. 
* train - Train the model using a dataset.
* backpropagation - Makes a forward pass and propagates back the gradient to update the weights during training.
* forward_pass - Makes a forward pass with the given weights and inputs.
* average_error - Calculates the average error of the model for a given dataset.
* get_error - Calculates the error for a given data point. The error function is chosen when initializing the NeuralNet class.
* predict - Predicts the classes of a set of datapoints
* score - Gets the percentage of correct predictions for a dataset
* squared_loss - The cost function implemented.

#HiddenLayer
Used as a superclass. Both FCLayer and ConvLayer classes inherit the activation functions from it.
* relu - The ReLU activation function.
* leaky_relu - The Leaky ReLU activation function.
* sigmoid - The Sigmoid activation function.
* tanh - The Tanh activation function.

#FCLayer
Fully Connected layer. Calculates the forward pass or the backward pass.

#ConvLayer
Concolution layer. Calculates the forward or the backward pass.

#OutputLayer
The output layer. It contains one node per class. There are 2 possible loss functions:
* square_loss - Let f be the value in a node corresponding to a false class and r be the value corresponding to the correct class. First we apply a ReLU to the value f-(r-1), so max(f-(r-1), 0), then we sum the square of these values, so it is the sum of (max(f-(r-1),0))^2 over each f.
* softmax_ce - The Softmax followed by the Cross Entropy function.
