# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)

        NOTE: The eval parameter is to indicate whether we are in the 
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """
        self.x = x

        if eval:
            
            self.norm = (self.x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            self.out = self.gamma * self.norm + self.beta
        else: 
            self.mean = np.mean(x,axis = 0)
            self.var = np.var(x, axis=0)
            self.norm = (self.x - self.mean) /np.sqrt(self.var + self.eps)
            self.out = self.gamma * self.norm + self.beta
            
            self.running_mean = self.running_mean * self.alpha + self.mean * (1 - self.alpha)
            self.running_var = self.running_var * self.alpha + self.var * (1 - self.alpha)

        return self.out


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        N = delta.shape[0]
        self.dgamma = np.sum(delta*self.norm,axis = 0,keepdims = True)
        self.dbeta = np.sum(delta, axis = 0, keepdims = True)
        
        dx = delta*self.gamma
        dvar = (-1/2) * np.sum(dx * (self.x-self.mean)*((self.var + self.eps)**(-3/2)), axis = 0)
        dmean = -np.sum(dx * ((self.var + self.eps)**(-1/2)), axis = 0)-(2/N) * dvar * np.sum(self.x-self.mean, axis = 0)

        p1 = dx * ((self.var + self.eps)**(-1/2))
        p2 = dvar*(2/N)*(self.x-self.mean)

        dx = p1+p2 + dmean/N

        return dx
