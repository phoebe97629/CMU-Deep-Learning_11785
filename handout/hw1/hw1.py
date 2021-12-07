#!/usr/bin/env python
# coding: utf-8

# In[4]:


"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *


# In[6]:


class MLP(object):

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.foo = [ bar(???) for ?? in ? ])
        # (HINT: Can you use zip here?)

        self.linear_layers = []
        # create a generic representation of the layers
        layers = [self.input_size] + hiddens + [self.output_size]

        for i, j in zip(layers[:-1], layers[1:]):
            self.linear_layers.append(Linear(i, j, weight_init_fn, bias_init_fn))

        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        if self.bn:
            self.bn_layers = []
            for i in range(self.num_bn_layers):
                self.bn_layers.append(BatchNorm(layers[i + 1]))

    def forward(self, x):
        # Complete the forward pass through your entire MLP.

        # Linear Layer → Batch Norm (if applicable) → Activation → Next Layer ...
        y = x
        # Note: the activation function will use during training.
        for i in range(self.nlayers):

            z = self.linear_layers[i].forward(y)

            if self.bn and i < self.num_bn_layers:
                if self.train_mode:
                    z = self.bn_layers[i].forward(z)
                else:
                    z = self.bn_layers[i].forward(z, eval=True)
            # update
            y = self.activations[i].forward(z)
        self.output = y
        return y

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        for i in range(len(self.linear_layers)):
            self.linear_layers[i].dW.fill(0.0)
            self.linear_layers[i].db.fill(0.0)

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers)
        if self.momentum:
            for i in range(len(self.linear_layers)):
                self.linear_layers[i].momentum_W = self.momentum * self.linear_layers[i].momentum_W - self.lr * \
                                                   self.linear_layers[i].dW
                self.linear_layers[i].momentum_B = self.momentum * self.linear_layers[i].momentum_B - self.lr * \
                                                   self.linear_layers[i].db
                ###UPDATED WEIGHT!!!!!
                self.linear_layers[i].W = self.linear_layers[i].W + self.linear_layers[i].momentum_W
                self.linear_layers[i].b = self.linear_layers[i].b + self.linear_layers[i].momentum_B

        else:
            for i in range(len(self.linear_layers)):
                self.linear_layers[i].W = self.linear_layers[i].W - self.lr * self.linear_layers[i].dW
                self.linear_layers[i].b = self.linear_layers[i].b - self.lr * self.linear_layers[i].db
            # Do the same for batchnorm layers
        if self.bn:
            for i in range(len(self.bn_layers)):
                self.bn_layers[i].gamma = self.bn_layers[i].gamma - self.lr * self.bn_layers[i].dgamma
                self.linear_layers[i].beta = self.linear_layers[i].beta - self.lr * bn_layers[i].dbeta

    def backward(self, labels):
        self.criterion.forward(self.output, labels)
        dy = self.criterion.derivative()
        for l in range(self.nlayers - 1, -1, -1):
            dz = dy * self.activations[l].derivative()
            if l >= self.num_bn_layers:
                dy = self.linear_layers[l].backward(dz)
            elif self.bn:
                dz = self.bn_layers[l].backward(dz)
                dy = self.linear_layers[l].backward(dz)


    def error(self, labels):
        return (np.argmax(self.output, axis=1) != np.argmax(labels, axis=1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


# In[7]:


#This function does not carry any points. You can try and complete this function to train your network.
def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...

    for e in range(nepochs):

        # Per epoch setup ...

        for b in range(0, len(trainx), batch_size):

            pass  # Remove this line when you start implementing this
            # Train ...

        for b in range(0, len(valx), batch_size):

            pass  # Remove this line when you start implementing this
            # Val ...

        # Accumulate data...

    # Cleanup ...

    # Return results ...

    # return (training_losses, training_errors, validation_losses, validation_errors)

    raise NotImplemented


# In[ ]:




