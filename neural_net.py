"""
TO DO
- Variable learning rate
- Batch Training
- Batch Normalization
- ADAM Gradient Descent
- Dropout

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

#%%

class NeuralNet:

    def __init__(self, n_inputs, n_outputs, output_layer='softmax_ce'):
        self.input_size = n_inputs
        self.output_size = n_outputs
        self.output_layer = output_layer
        #We will store all the layers directly in this list.
        self.loss_function = output_layer
        self.hidden_layers = []
        self.output_layer = OutputLayer(n_inputs, n_outputs, output_layer)

    def add_layer(self, n_nodes, activation='ReLU'):
        if not self.hidden_layers:
            self.hidden_layers.append(HiddenLayer(self.input_size, 
                                                  n_nodes, activation))
        else:
            self.hidden_layers.append(HiddenLayer(self.hidden_layers[-1].n_nodes, 
                                                  n_nodes, activation))
        self.output_layer = OutputLayer(self.hidden_layers[-1].n_nodes, 
                                        self.output_size, self.loss_function)
            
    def train(self, features, targets, epochs=100, 
              learning_rate=0.001, momentum=False):
        weight_grads = [np.zeros((layer.weights.shape[0],layer.weights.shape[1]+1))
                                 for layer in self.hidden_layers]
        weight_grads.append(np.zeros((self.output_layer.weights.shape[0],
                                      self.output_layer.weights.shape[1]+1)))
        if momentum:
            rho = 0.9
            velocities = weight_grads.copy()
        for epoch in range(epochs):
            for feature, target in zip(features, targets):
                #Backpropagates the error to update weights
                new_grads, _ = self.backpropropagation(feature, target)
                weight_grads = [weight_grads[i]+new_grads[i]
                                for i in range(len(weight_grads))]
            weight_grads = [grad/len(targets) for grad in weight_grads] 
            #Updates the weights of the hidden layers
            if momentum:
                velocities = rho*velocities + weight_grads
                weight_grads = velocities.copy()
            self.update_weights(weight_grads, learning_rate, momentum)
            
    def plot_train(self, features, targets, test_features, test_targets,
                   epochs=100, learning_rate=0.1, momentum=False):
        train_error = np.zeros(epochs)
        test_error = np.zeros(epochs)
        weight_grads = [np.zeros((layer.weights.shape[0],layer.weights.shape[1]+1))
                                 for layer in self.hidden_layers]
        weight_grads.append(np.zeros((self.output_layer.weights.shape[0],
                                      self.output_layer.weights.shape[1]+1)))
        if momentum:
            rho = 0.9
            velocities = weight_grads.copy()
        for epoch in range(epochs):
            for feature, target in zip(features, targets):
                #Backpropagates the error to update weights
                new_grads, error = self.backpropropagation(feature, target)
                train_error[epoch] += error
                weight_grads = [weight_grads[i]+new_grads[i]
                                for i in range(len(weight_grads))]
            weight_grads = [grad/len(targets) for grad in weight_grads] 
            #Updates the weights of the hidden layers
            for index, layer in enumerate(self.hidden_layers):
                grad = weight_grads[index]
                #Updates the biases
                layer.bias = layer.bias - (learning_rate*grad[:,0])
                #Updates the weights in each backpropagation.
                layer.weights = layer.weights-(learning_rate*grad[:,1:])
            if momentum:
                velocities = rho*velocities + weight_grads
                weight_grads = velocities.copy()
            train_error[epoch] = train_error[epoch]/len(targets)
            test_error[epoch] = self.average_error(test_features, test_targets)
            self.update_weights(weight_grads, learning_rate, momentum)
        fig = plt.figure()
        ax = fig.add_axes([0,0,2,1])
        ax.plot(train_error, label='Train')
        ax.plot(test_error, label='Test')
        ax.set_title('Test x Train error')
        ax.legend()
        plt.show()

    def update_weights(self, weight_grads, learning_rate, momentum):
        for index, layer in enumerate(self.hidden_layers):
            grad = weight_grads[index]
            #Updates the biases
            layer.bias = layer.bias - (learning_rate*grad[:,0])
            #Updates the weights in each backpropagation.
            layer.weights = layer.weights-(learning_rate*grad[:,1:])
        #Updates the weights of the output layer
        grad = weight_grads[-1]
        #Updates the biases
        self.output_layer.bias = (self.output_layer.bias - 
                                  (learning_rate*grad[:,0]))
        #Updates the weights in each backpropagation.
        self.output_layer.weights = (self.output_layer.weights-
                                     (learning_rate*grad[:,1:]))
            
    def backpropropagation(self, feature, target):
        #Gets the gradient of the error
        error = self.forward_pass(feature, target)
        node_grad, weight_grad = self.output_layer.backwards(feature, target)
        weight_grads = [weight_grad]
        #Loops backwards through all layers
        for layer in self.hidden_layers[::-1]:
            #Feeds the previous gradient to the next layer
            node_grad, weight_grad = layer.backwards(node_grad)
            #Stores the gradients with respect to the weights so they can be 
            #averaged after the end of an epoch
            weight_grads.append(weight_grad)
        return(weight_grads[::-1], error)

    #Calculates the outputs for a given input
    def forward_pass(self, inputs, target):
        #Calculates the outputs of each layer and feed that as inputs for the 
        #next
        for layer in self.hidden_layers:
            inputs = layer.forwards(inputs)
        error = self.output_layer.get_error(inputs, target)
        return(error)

    def average_error(self, features, target_indexes):
        #Calculates the average error in a dataset
        n_targets = len(target_indexes)
        if n_targets == 1:
            return(self.forward_pass(features, target_indexes))
        else:
            error = 0
            for i in range(n_targets):
                error += self.forward_pass(features[i,:], target_indexes[i])
            return(error/n_targets)

    def predict(self, features):
    #Prdicts the correct output for a given input
        prediction = []
        for feature in features:
            prediction.append(np.argmax(self.forward_pass(feature, -1)))
        return(prediction)
    
    def score(self, features, targets):
        #Prints the percentage of correct predictions for this dataset
        prediction = self.predict(features)
        score = np.sum(np.equal(prediction, targets))*100/len(targets)     
        return(score)
            
#%%

class HiddenLayer:
    
    #Initialize a hidden layer with a given ammount of nodes and inputs
    def __init__(self, n_inputs, n_nodes, activation):
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        self.weights = (np.random.random((n_nodes, n_inputs))*2)-1
        self.bias = (np.random.random(n_nodes)*2)-1
        #Possible activation functions: ReLu, LeakyReLu, Tanh and Sigmoid
        #We also have a sofmax activation for the output layer
        activation = activation.lower()
        if activation == 'relu':
            self.activation = self.relu
        elif activation == 'leaky_relu':
            self.activation = self.leaky_relu            
        elif activation == 'tanh':
            self.activation = self.tanh
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
        else:
            raise Exception('No such activation function.')
            
    def forwards(self, inputs):
        #We need to store these values so we can use then for backpropagation
        self.inputs = inputs
        #Multiplies the inputs by the respective weights
        outputs = self.weights.dot(inputs) + self.bias
        #Go through the activation function 
        outputs = self.activation(outputs)
        return(outputs)
    
    def backwards(self, grad_in):
        #Corresponds to the activation function
        grad_in = self.activation(grad_in, forward=False)
        #Calculates the gradient for each node to be passed to the next layer
        node_grad = (self.weights.T).dot(grad_in)
        #Calculates the gradient for each weight and the biases
        grad_in = np.array([grad_in,] * (self.n_inputs+1)).T
        inputs = np.array([[1,*self.inputs],] * self.n_nodes)
        inputs[:,0] = 1
        weight_grad = grad_in * inputs
        return(node_grad, weight_grad)

#_______________________________________________#
    #Here are stored the activation functions    
    def relu(self, inputs, forward=True):
        if forward:
            outputs = np.maximum(0, inputs)
            #Value needed to backpropagate
            self.store = np.array(outputs>0, int)
            return(outputs)
        else:
            return(inputs*self.store)
        
    def leaky_relu(self, inputs, forward=True):
        if forward == True:
            #Store for backpropagation
            self.store = np.where(inputs<0, 0.01, 1)
            outputs = inputs*self.store
            return(outputs)
        else:
            return(inputs*self.store)

    def sigmoid(self, inputs, forward=True):
        if forward == True:
            outputs = 1/(1+np.exp(inputs))
            return(outputs)
        else:
            sigmoid = 1/(1+np.exp(inputs))
            return(sigmoid*(1-sigmoid))
        
    def tanh(self, inputs, forward=True):
        if forward == True:
            return(np.tanh(inputs))
        else:
            return(1-(np.tanh(inputs)**2))

class OutputLayer:

     #Initialize the output layer with a given ammount of nodes and inputs
    def __init__(self, n_inputs, n_nodes, activation):
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        self.weights = (np.random.random((n_nodes, n_inputs))*2)-1
        self.bias = (np.random.random(n_nodes)*2)-1
        #Possible activation functions: ReLu, LeakyReLu, Tanh and Sigmoid
        #We also have a sofmax activation for the output layer
        activation = activation.lower()
        if activation == 'softmax_ce':
            self.activation = self.softmax_ce
        elif activation == 'squared_loss':
            self.activation = self.squared_loss
        else:
            raise Exception('No such loss function.')
            
    def get_output(self, inputs):
        #We need to store these values so we can use then for backpropagation
        self.inputs = inputs
        #Multiplies the inputs by the respective weights
        outputs = self.weights.dot(inputs) + self.bias
        #Go through the activation function 
        outputs = self.activation(outputs, -1)
        return(outputs)

    def get_error(self, inputs, target):
        #We need to store these values so we can use then for backpropagation
        self.inputs = inputs
        #Multiplies the inputs by the respective weights
        outputs = self.weights.dot(inputs) + self.bias
        #Go through the activation function 
        outputs = self.activation(outputs, target)
        return(outputs)

    def backwards(self, grad_in, target):
        #Corresponds to the activation function
        grad_in = self.activation(grad_in, target, True)
        #Calculates the gradient for each node to be passed to the next layer
        node_grad = (self.weights.T).dot(grad_in)
        #Calculates the gradient for each weight and the biases
        grad_in = np.array([grad_in,] * (self.n_inputs+1)).T
        inputs = np.array([[1,*self.inputs],] * self.n_nodes)
        inputs[:,0] = 1
        weight_grad = grad_in * inputs
        return(node_grad, weight_grad)
#______________________________________________________________#        
    #Here are stored the loss functions for the output layer
    def squared_loss(self, inputs, target_index, backprop=False):
        #Let o_0, o_1, o_2, etc be the output values of the model and let k be
        #the index of the correct answer. Suppose n different from k.
        #If o_n < o_k -1, then the error for this output is 0. If o_n < o_k-1,
        #then the penalty is (o_n-o_k)^2. We then sum all these penalties for
        #all o_n with n !+ k.
        #In words, we do not penalize if a wrong output value is below a safe
        #margin of the correct value, we penalize only a little if it is below
        #the value of the correct output, but not below the margin, and we
        #penalize a lot if it is above the value of the correct output.
        if backprop:
            grad = self.store
            grad = grad*2
            grad[target_index] = -np.sum(grad)
            return(grad)
        else:            
            outputs = (inputs-inputs[target_index])+1
            outputs[target_index] = 0
            outputs = np.maximum(outputs, 0)
            self.store = outputs
            outputs = outputs**2
            #If target_index is -1, then we do not know what is the cooret
            #class. Used for predicting
            if target_index == -1:
                return(outputs)
            else:
                return((np.sum(outputs)/len(inputs)))

    def softmax_ce(self, inputs, target_index, backprop=False):
        if backprop:
            grad = self.store
            grad[target_index] -= 1    
            return(grad)
        # grad = np.array([self.store,]*self.n_nodes)
        # grad = grad*grad.T
        # grad = grad - (grad*np.eye(self.n_nodes))
        # grad = grad*np.array([inputs,]*self.n_nodes)
        # return(np.sum(grad, axis=0))
        else:
            cnst = np.max(inputs)
            inputs = inputs-cnst
            outputs = np.exp(inputs)
            outputs = outputs/np.sum(outputs)
            #This is the output before calculating the error, so it is the
            #scores for each of the classes
            self.store = outputs
            #If target_index is -1, then we do not know what is the cooret
            #class. Used for predicting
            if target_index == -1:
                return(outputs)
            else:
                #Returns the cost, the gradient and the outputs before cost
                return(-1*np.log(outputs[target_index]))

        
#%%

#This is a small demonstration using the iris dataset
if __name__ == '__main__':
    np.set_printoptions(precision=3)
    #Loading the Iris dataset
    iris = datasets.load_iris()
    features = iris.data
    targets = iris.target
    #Makes the dataset zero-mean and unit variant
    std = np.std(features, axis=0)
    avg = np.mean(features, axis=0)
    features = (features-avg)/std
    #Shuffle the features and targets
    permutation = np.random.permutation(len(targets))
    targets = targets[permutation]
    features = features[permutation,:]
    #Separate dataset in test and train
    test_X = features[100:,:]
    test_Y = targets[100:]
    train_X = features[:100,:]
    train_Y = targets[:100]
    
    #Build the neural network
    model = NeuralNet(4, 3, 'softmax_ce')
    model.add_layer(10, activation='leaky_relu')
    model.add_layer(10, activation='leaky_relu')
    model.add_layer(10, activation='leaky_relu')
    
    #Train the model
    # model.train(train_X, train_Y, learning_rate = 0.001)
    model.plot_train(train_X, train_Y, test_X, test_Y, 
                      learning_rate=0.05, epochs=100)
    
    #Get the score
    score = model.score(test_X, test_Y)
    print(score)
