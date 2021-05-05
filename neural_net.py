"""
Created on Thu Apr 22 13:56:31 2021

@author: kling

TO DO
- Variable learning rate
- Batch Training
- Batch Normalization
- ADAM Gradient Descent
- Dropout
- Review Train Method
- Implement stride and padding for ConvLayer
- Optimize ConvLayer computations. Eliminate for loops.

"""

import numpy as np
import matplotlib.pyplot as plt

#%%

class NeuralNet:

    def __init__(self, input_shape, n_outputs, output_layer='softmax_ce'):
        self.input_shape = input_shape
        if len(input_shape)==1:
            self.input_size = input_shape
        elif len(input_shape)==2:
            self.input_size = input_shape[0]*input_shape[1]
        elif len(input_shape)==3:
            self.input_size = input_shape[0]*input_shape[1]*input_shape[2]
        else:
            raise Exception('Invalid input shape.')
        self.output_size = n_outputs
        self.output_layer = output_layer
        #We will store all the layers directly in this list.
        self.loss_function = output_layer
        self.hidden_layers = []
        self.output_layer = OutputLayer(self.input_size, n_outputs, output_layer)

    def add_FC_layer(self, n_nodes, activation='ReLU'):
        if not self.hidden_layers:
            self.hidden_layers.append(FCLayer(self.input_size, 
                                                  n_nodes, activation))
        else:
            self.hidden_layers.append(FCLayer(self.hidden_layers[-1].n_nodes, 
                                                  n_nodes, activation))
        self.output_layer = OutputLayer(self.hidden_layers[-1].n_nodes, 
                                        self.output_size, self.loss_function)
        
    def add_Conv_layer(self, filter_size, depth, 
                       stride, padding, activation='ReLU'):
        if not self.hidden_layers:
            self.hidden_layers.append(ConvLayer(self.input_shape, filter_size,
                                                depth, stride, padding, activation))
        else:
            if isinstance(self.hidden_layers[-1], FCLayer):
                raise Exception('Cannot use FCLayer before a ConvLayer')
            self.hidden_layers.append(ConvLayer(self.hidden_layers[-1].output_shape, 
                                                filter_size, depth, stride, 
                                                padding, activation))
        self.output_layer = OutputLayer(self.hidden_layers[-1].n_nodes, 
                                        self.output_size, self.loss_function)
            
    def train(self, features, targets, epochs=100, 
              learning_rate=0.001, momentum=False):
        weight_grads = [np.zeros_like(layer.weights)
                                 for layer in self.hidden_layers]
        weight_grads.append(np.zeros_like(self.output_layer.weights))
        bias_grads = [np.zeros_like(layer.bias)
                                 for layer in self.hidden_layers]
        bias_grads.append(np.zeros_like(self.output_layer.bias))
        if momentum:
            rho = 0.9
            w_velocities = weight_grads.copy()
            b_velocities = bias_grads.copy()
        for epoch in range(epochs):
            print('Epoch:', epoch)
            for feature, target in zip(features, targets):
                #Backpropagates the error to update weights
                new_w_grads, new_b_grads, _ = self.backpropropagation(feature, 
                                                                      target)
                weight_grads = [weight_grads[i]+new_w_grads[i]
                                for i in range(len(weight_grads))]
                bias_grads = [bias_grads[i]+new_b_grads[i]
                                for i in range(len(bias_grads))]
            weight_grads = [grad/len(targets) for grad in weight_grads]
            bias_grads = [grad/len(targets) for grad in bias_grads]
            #Updates the weights of the hidden layers
            if momentum:
                w_velocities = [rho*velocity + weight_grad for 
                        velocity, weight_grad in zip(w_velocities, weight_grads)]
                weight_grads = w_velocities.copy()
                b_velocities = [rho*velocity + bias_grad for 
                        velocity, bias_grad in zip(b_velocities, bias_grads)]
                bias_grads = b_velocities.copy()
            self.update_weights(weight_grads, bias_grads, learning_rate, momentum)
            
    def plot_train(self, features, targets, test_features, test_targets,
                   epochs=100, learning_rate=0.1, momentum=False):
 
        train_error = np.zeros(epochs)
        test_error = np.zeros(epochs)
        weight_grads = [np.zeros_like(layer.weights)
                                 for layer in self.hidden_layers]
        weight_grads.append(np.zeros_like(self.output_layer.weights))
        bias_grads = [np.zeros_like(layer.bias)
                                 for layer in self.hidden_layers]
        bias_grads.append(np.zeros_like(self.output_layer.bias))
        if momentum:
            rho = 0.9
            w_velocities = weight_grads.copy()
            b_velocities = bias_grads.copy()
        for epoch in range(epochs):
            print('Epoch:', epoch)
            for feature, target in zip(features, targets):
                #Backpropagates the error to update weights
                new_w_grads, new_b_grads, error = self.backpropropagation(feature, 
                                                                      target)
                train_error[epoch] += error
                weight_grads = [weight_grads[i]+new_w_grads[i]
                                for i in range(len(weight_grads))]
                bias_grads = [bias_grads[i]+new_b_grads[i]
                                for i in range(len(bias_grads))]
            weight_grads = [grad/len(targets) for grad in weight_grads]
            bias_grads = [grad/len(targets) for grad in bias_grads]
            #Updates the weights of the hidden layers
            if momentum:
                w_velocities = [rho*velocity + weight_grad for 
                        velocity, weight_grad in zip(w_velocities, weight_grads)]
                weight_grads = w_velocities.copy()
                b_velocities = [rho*velocity + bias_grad for 
                        velocity, bias_grad in zip(b_velocities, bias_grads)]
                bias_grads = b_velocities.copy()
            train_error[epoch] = train_error[epoch]/len(targets)
            test_error[epoch] = self.average_error(test_features, test_targets)
            self.update_weights(weight_grads, bias_grads, learning_rate, momentum)
        fig = plt.figure()
        ax = fig.add_axes([0,0,2,1])
        ax.plot(train_error, label='Train')
        ax.plot(test_error, label='Test')
        ax.set_title('Test x Train error')
        ax.legend()
        plt.show()

    def update_weights(self, weight_grads, bias_grads, learning_rate, momentum):
        for index, layer in enumerate(self.hidden_layers):
            w_grad = weight_grads[index]
            b_grad = bias_grads[index]
            #Updates the biases
            layer.bias = layer.bias - (learning_rate*b_grad)
            #Updates the weights in each backpropagation.
            layer.weights = layer.weights-(learning_rate*w_grad)
        #Updates the weights of the output layer
        w_grad = weight_grads[-1]
        b_grad = bias_grads[-1]
        #Updates the biases
        self.output_layer.bias = (self.output_layer.bias - 
                                  (learning_rate*b_grad))
        #Updates the weights in each backpropagation.
        self.output_layer.weights = (self.output_layer.weights-
                                     (learning_rate*w_grad))
            
    def backpropropagation(self, feature, target):
        #Gets the gradient of the error
        error = self.forward_pass(feature, target)
        node_grad, weight_grad, bias_grad = self.output_layer.backwards(feature, target)
        weight_grads = [weight_grad]
        bias_grads = [bias_grad]
        #Loops backwards through all layers
        for layer in self.hidden_layers[::-1]:
            #Feeds the previous gradient to the next layer
            node_grad, weight_grad, bias_grad = layer.backwards(node_grad)
            #Stores the gradients with respect to the weights so they can be 
            #averaged after the end of an epoch
            weight_grads.append(weight_grad)
            bias_grads.append(bias_grad)
        return(weight_grads[::-1], bias_grads[::-1], error)

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
    
    def set_activation(self, activation):
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

#%%
        
class FCLayer(HiddenLayer):
    
    #Initialize a hidden layer with a given ammount of nodes and inputs
    def __init__(self, n_inputs, n_nodes, activation):
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        self.weights = (np.random.random((n_nodes, n_inputs))*2)-1
        self.bias = (np.random.random(n_nodes)*2)-1
        super().set_activation(activation)
            
    def forwards(self, inputs):
        inputs = inputs.flatten()
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
        #Gradients of the biases
        bias_grads = grad_in
        #Calculates the gradient for each weight and the biases
        grad_in = np.array([grad_in,] * self.n_inputs).T
        inputs = np.array([self.inputs,] * self.n_nodes)
        weight_grad = grad_in * inputs
        return(node_grad, weight_grad, bias_grads)

#%%

class ConvLayer(HiddenLayer):
    
    def __init__(self, input_shape, filter_size, depth, 
                 stride, padding, activation):
        self.input_shape = input_shape
        self.filter_size = filter_size
        self.depth = depth
        self.stride = stride
        self.padding = padding
        self.output_shape = (input_shape[0]-filter_size+1, 
                             input_shape[1]-filter_size+1,
                             depth) #No padding nor stride
        self.n_nodes = (self.output_shape[0]*
                        self.output_shape[1]*
                        self.output_shape[2])
        if len(input_shape)==2:
            self.d2 = True
            self.input_shape = (input_shape[0], input_shape[1])
            self.weights = (np.random.random((filter_size, filter_size, 
                                              depth))*2)-1
        elif len(input_shape)==3:
            self.d2 = False
            self.weights = (np.random.random((filter_size, filter_size,
                                              input_shape[2], depth))*2)-1
        else:
            raise Exception('Invalid input shape for Convolutional layer')
        self.bias = (np.random.random(depth)*2)-1
        super().set_activation(activation)        

    def forwards(self, inputs):
        self.store = inputs
        fs = self.filter_size
        height = self.input_shape[0]
        width = self.input_shape[1]
        output = np.zeros(self.output_shape)
        if self.d2:
            for d in range(self.depth):
                for i in range(height-fs+1):
                    for j in range(width-fs+1):
                      output[i,j,d] = np.sum(inputs[i:i+fs,j:j+fs]*
                                             self.weights[:,:,d])+self.bias[d]
        else:
            for d in range(self.depth):
                for i in range(height-fs+1):
                    for j in range(width-fs+1):
                      output[i,j,d] = np.sum(inputs[i:i+fs,j:j+fs]*
                                             self.weights[:,:,:,d])+self.bias[d]
        return(output)
    
    def backwards(self, grad_in):
        grad_in = grad_in.reshape(self.output_shape)
        fs = self.filter_size
        height = self.input_shape[0]
        width = self.input_shape[1]
        weight_grads = np.zeros_like(self.weights)
        node_grads = np.zeros(self.input_shape)
        bias_grads = np.zeros(self.depth)
        if self.d2:
            for d in range(self.depth):
                bias_grads[d] = np.sum(self.store)
                for i in range(height-fs):
                    for j in range(width-fs):
                        temp_w = self.store[i:i+fs, j:j+fs]*grad_in[i,j,d]
                        temp_n = weight_grads[:,:,d]*grad_in[i,j,d]
                        weight_grads[:,:,d] += temp_w
                        node_grads[i:i+fs, j:j+fs] += temp_n
        else:
            for d in range(self.depth):
                bias_grads[d] = np.sum(self.store)
                for i in range(height-fs):
                    for j in range(width-fs):
                        temp_w = self.store[i:i+fs, j:j+fs]*grad_in[i,j,d]
                        temp_n = weight_grads[:,:,:,d]*grad_in[i,j,d]
                        weight_grads[:,:,:,d] += temp_w
                        node_grads[i:i+fs, j:j+fs, :] += temp_n
        return(node_grads, weight_grads, bias_grads)

#%%

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
        inputs = inputs.flatten()
        #We need to store these values so we can use then for backpropagation
        self.inputs = inputs
        #Multiplies the inputs by the respective weights
        outputs = self.weights.dot(inputs) + self.bias
        #Go through the activation function 
        outputs = self.activation(outputs, -1)
        return(outputs)

    def get_error(self, inputs, target):
        inputs = inputs.flatten()
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
        #Gradients of the biases
        bias_grads = grad_in
        #Calculates the gradient for each node to be passed to the next layer
        node_grad = (self.weights.T).dot(grad_in)
        #Calculates the gradient for each weight and the biases
        grad_in = np.array([grad_in,] * self.n_inputs).T
        inputs = np.array([self.inputs,] * self.n_nodes)
        weight_grad = grad_in * inputs
        return(node_grad, weight_grad, bias_grads)
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
        else:
            cnst = np.max(inputs)
            inputs = inputs-cnst
            outputs = np.exp(inputs)
            outputs = outputs/np.sum(outputs)
            #This is the output before calculating the error, so it is the
            #scores for each of the classes
            self.store = outputs
            #If target_index is -1, then we do not know what is the correct
            #class. Used for predicting
            if target_index == -1:
                return(outputs)
            else:
                #Returns the cost
                return(-1*np.log(outputs[target_index]))

        
#%%

#This is a small demonstration using the iris dataset
if __name__ == '__main__':
    # np.set_printoptions(precision=3)
    # #Loading the Iris dataset
    # from sklearn.datasets import load_iris
    # iris = load_iris()
    # features = iris.data
    # targets = iris.target
    # #Makes the dataset zero-mean and unit variant
    # std = np.std(features, axis=0)
    # avg = np.mean(features, axis=0)
    # features = (features-avg)/std
    # #Shuffle the features and targets
    # permutation = np.random.permutation(len(targets))
    # targets = targets[permutation]
    # features = features[permutation,:]
    # #Separate dataset in test and train
    # test_X = features[100:,:]
    # test_Y = targets[100:]
    # train_X = features[:100,:]
    # train_Y = targets[:100]
    
    # #Build the neural network
    # model = NeuralNet(4, 3, 'softmax_ce')
    # model.add_layer(10, activation='leaky_relu')
    # model.add_layer(10, activation='leaky_relu')
    # model.add_layer(10, activation='leaky_relu')
    
    # #Train the model
    # model.train(train_X, train_Y, learning_rate = 0.01, momentum=True)
    # # model.plot_train(train_X, train_Y, test_X, test_Y, 
    # #                   learning_rate=0.05, epochs=100, momentum=True)
    
    # #Get the score
    # score = model.score(test_X, test_Y)
    # print(score)
    
    from sklearn.datasets import load_digits

    digits = load_digits( )
    np.set_printoptions(precision=3)
    #Loading the Iris dataset
    features = digits.data
    targets = digits.target
    features = features/16
    #Makes the data zero mean
    avg = np.mean(features, axis=0)
    features = (features-avg)
    #Shuffle the features and targets
    permutation = np.random.permutation(len(targets))
    targets = targets[permutation]
    features = features[permutation,:]
    features = features.reshape((features.shape[0], 8, 8))
    #Separate dataset in test and train
    test_X = features[1200:,:]
    test_Y = targets[1200:]
    train_X = features[:1200,:]
    train_Y = targets[:1200]

    #Build the neural network
    model = NeuralNet((8,8), 10, 'softmax_ce')
    model.add_Conv_layer(3,20,1,1,'relu')
    model.add_FC_layer(20, activation='relu')
    
    #Train the model
    model.train(train_X, train_Y, learning_rate = 0.005, epochs=200, momentum=True)
    # model.plot_train(train_X, train_Y, test_X, test_Y, 
    #                   learning_rate=0.05, epochs=50, momentum=True)
    print(model.score(test_X, test_Y))
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
