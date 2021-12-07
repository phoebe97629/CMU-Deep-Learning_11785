# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

"""
In the linear.py file, attributes have been added to the Linear class to make
implementing Adam easier, check them out!

self.mW = np.zeros(None) #mean derivative for W
self.vW = np.zeros(None) #squared derivative for W
self.mb = np.zeros(None) #mean derivative for b
self.vb = np.zeros(None) #squared derivative for b
"""

class adam():
    def __init__(self, model, beta1=0.9, beta2=0.999, eps = 1e-8):
        self.model = model
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = self.model.lr
        self.t = 0 # Number of Updates

    def step(self):
        '''
        * self.model is an instance of your MLP in hw1/hw1.py, it has access to
          the linear layer's list.
        * Each linear layer is an instance of the Linear class, and thus has
          access to the added class attributes dicussed above as well as the
          original attributes such as the weights and their gradients.
        '''
        self.t += 1

        # Add your code here!
        #gt is the current gradient
        for l in self.model.linear_layers:
            l.mW = self.beta1 * l.mW + (1 - self.beta1)*l.dW
            l.vW = self.beta2 * l.vW + (1 - self.beta2) * l.dW**2

            l.mb = self.beta1 * l.mb + (1 - self.beta1) * l.db
            l.vb = self.beta2 * l.vb + (1 - self.beta2) * l.db**2

            mW = l.mW / (1 - self.beta1 ** self.t)
            mb = l.mb / (1 - self.beta1 ** self.t)
            vW = l.vW / (1 - self.beta2 ** self.t)
            vb = l.vb / (1 - self.beta2 ** self.t)

            l.W -= self.lr * mW / np.sqrt(vW + self.eps)
            l.b -= self.lr * mb / np.sqrt(vb + self.eps)


            #return







class adamW():
    def __init__(self, model, beta1=0.9, beta2=0.999, eps = 1e-8, weight_decay = 0.01):
        self.model = model
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = self.model.lr
        self.t = 0 # Number of Updates
        self.weight_decay  = weight_decay

    def step(self):
        '''
        * self.model is an instance of your MLP in hw1/hw1.py, it has access to
          the linear layer's list.
        * Each linear layer is an instance of the Linear class, and thus has
          access to the added class attributes dicussed above as well as the
          original attributes such as the weights and their gradients.
        '''

        #self.mW = np.zeros(None)  # mean derivative for W
        #self.vW = np.zeros(None)  # squared derivative for W
        #self.mb = np.zeros(None)  # mean derivative for b
        #self.vb = np.zeros(None)  # squared derivative for b


        self.t += 1

        # Add your code here!
        #gt is the current gradient
        for l in self.model.linear_layers:
            l.mW = self.beta1 * l.mW + (1 - self.beta1) * l.dW
            l.vW = self.beta2 * l.vW + (1 - self.beta2) * l.dW ** 2

            l.mb = self.beta1 * l.mb + (1 - self.beta1) * l.db
            l.vb = self.beta2 * l.vb + (1 - self.beta2) * l.db ** 2

            mW = l.mW / (1 - self.beta1 ** self.t)
            mb = l.mb / (1 - self.beta1 ** self.t)
            vW = l.vW / (1 - self.beta2 ** self.t)
            vb = l.vb / (1 - self.beta2 ** self.t)

            l.W -= self.lr * (mW / np.sqrt(vW + self.eps) + self.weight_decay*l.W)
            l.b -= self.lr * (mb / np.sqrt(vb + self.eps) + self.weight_decay*l.b)


