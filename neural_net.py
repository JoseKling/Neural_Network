#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 13:56:31 2021

@author: kling
"""

import numpy as np
from sklearn import datasets

#%%

class NeuralNet:

    def __init__(self, n_inputs, n_outputs, cost_function='squared_loss'):
        self.input_size = n_inputs
        self.output_size = n_outputs
        #We will store all the layers directly in this list.
        self.layers = []
        cost_function = cost_function.lower()
        if cost_function == 'cross_entropy':
            self.error_function = self.cross_entropy
        elif cost_function.lower() == 'squared_loss':
            self.error_function = self.squared_loss
        else:
            raise Exception('There is no such cost function.')

    def add_layer(self, n_nodes, activation='ReLU', output_layer=False):
        #If this is the first hidden layer, than the inputs of this node are 
        #the inputs of the network
        if not self.layers:
            self.layers.append(Layer(self.input_size, 
                                           n_nodes, activation))
        #If this is the last layer, number of nodes is the output size
        elif output_layer:
            self.layers.append(Layer(self.layers[-1].n_nodes, 
                                           self.output_size, activation))
        #Otherwise, its inputs are the outputs of the previous layer
        else:
            self.layers.append(Layer(self.layers[-1].n_nodes, 
                                           n_nodes, activation))
            
    def train(self, features, targets, epochs=100, 
              learning_rate=0.1, print_error=False):
        weight_grads = [np.zeros((layer.weights.shape[0],layer.weights.shape[1]+1))
                                 for layer in self.layers]
        for epoch in range(epochs):
            for feature, target in zip(features, targets):
                #Backpropagates the error to update weights
                new_grads = self.backpropropagation(feature, target)
                weight_grads = [weight_grads[i]+new_grads[i]
                                for i in range(len(weight_grads))]
            weight_grads = [grad/len(targets) for grad in weight_grads] 
            for index, layer in enumerate(self.layers):
                grad = weight_grads[index]
                #Updates the biases
                layer.bias = layer.bias - (learning_rate*grad[:,0])
                #Updates the weights in each backpropagation. Used only in online 
                #learning
                layer.weights = layer.weights-(learning_rate*grad[:,1:])
            if print_error:
                print('Total error for epoch {}:'.format(epoch), 
                      self.average_error(features, targets))
            
    def backpropropagation(self, feature, target):
        #Gets the gradient of the error
        node_grad = self.get_error(feature, target, True)
        weight_grads = []
        #Loops backwards through all layers
        for layer in self.layers[::-1]:
            #Feeds the previous gradient to the next layer
            node_grad, weight_grad = layer.backwards(node_grad)
            #Stores the gradients with respect to the weights so they can be 
            #averaged after the end of an epoch
            weight_grads.append(weight_grad)
        return(weight_grads[::-1])

    #Calculates the outputs for a given input
    def forward_pass(self, inputs):
        #Calculates the outputs of each layer and feed that as inputs for the 
        #next
        for layer in self.layers:
            inputs = layer.forwards(inputs)
        return(inputs)

    def average_error(self, features, target_indexes):
        #Calculates the average error in a dataset
        n_targets = len(target_indexes)
        if n_targets == 1:
            return(self.get_error(features, target_indexes))
        else:
            error = 0
            for i in range(n_targets):
                error += self.get_error(features[i], target_indexes[i])
            return(error/n_targets)

    #We need to define some error function in order to be able to train the 
    #network through backpropagation.
    #Answer should be the index of the correct answer in the outputs array
    def get_error(self, feature, target_index, backprop=False):
        output = self.forward_pass(feature)
        #If backprop is True, then it returns the gradient of the error
        #function, and not the error itself
        error = self.error_function(output, target_index, backprop)
        return(error)
    
    def predict(self, features):
    #Prdicts the correct output for a given input
        prediction = []
        for feature in features:
            prediction.append(np.argmax(self.forward_pass(feature)))
        return(prediction)
    
    def score(self, features, targets):
        #Prints the percentage of correct predictions for this dataset
        prediction = self.predict(features)
        score = np.sum(np.equal(prediction, targets))*100/len(targets)     
        print('Score: {:.2f}%'.format(score))

    #Here are stored the loss functions
    def squared_loss(self, output, target_index, backprop=False):
        #Let o_0, o_1, o_2, etc be the output values of the model and let k be
        #the index of the correct answer. Suppose n different from k.
        #If o_n < o_k -1, then the error for this output is 0. If o_n < o_k-1,
        #then the penalty is (o_n-o_k)^2. We then sum all these penalties for
        #all o_n with n !+ k.
        #In words, we do not penalize if a wrong output value is below a safe
        #margin of the correct value, we penalize only a little if it is below
        #the value of the correct output, but not below the margin, and we
        #penalize a lot if it is above the value of the correct output. 
        error = (output-output[target_index])+1
        error[target_index] = 0
        error = np.maximum(error, 0)
        if backprop:
            grad = error*2
            grad[target_index] = -np.sum(grad)
            return(grad)
        else:
            error = error**2
            error = (np.sum(error))/self.output_size
            return(error)

    def cross_entropy(self, output, target_index):
        pass

#%%

class Layer:
    
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
        elif activation == 'leakyrelu':
            self.activation = self.leaky_relu            
        elif activation == 'tanh':
            self.activation = self.tanh
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
        elif activation == 'softmax':
            self.activation = self.softmax            
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

    #Here are stored the activation functions    
    def relu(self, inputs, forward=True):
        if forward == True:
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
        
    def softmax(self, inputs, forward=True):
        if forward == True:
            cnst = np.max(inputs)
            inputs = inputs-cnst
            outputs = np.exp(inputs)
            outputs = outputs/np.sum(outputs)
            self.store = outputs
            return(outputs)
        else:
            grad = np.array([self.store,]*self.n_nodes)
            grad = grad*grad.T
            grad = grad - (grad*np.eye(self.n_nodes))
            grad = grad*np.array([inputs,]*self.n_nodes)
            return(np.sum(grad, axis=0))
        
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
    model = NeuralNet(4, 3)
    model.add_layer(10, activation='relu')
    model.add_layer(5, activation='softmax', output_layer=True)
    
    #Train the model
    model.train(train_X, train_Y, learning_rate=0.1, 
                epochs=100, print_error=True)
    #Get the score
    model.score(test_X, test_Y)