# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

#Each neurons has a probability p of being omitted.
#Not dropout at output layers.


import numpy as np


class Dropout2d(object):
    def __init__(self, p=0.5):
        # Dropout probability
        self.p = p

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, train=True):
        """
        Arguments:
          x (np.array): (batch_size, in_channel, input_width, input_height)
          train (boolean): whether the model is in training mode
        Return:
          np.array of same shape as input x
        """
        # 1) Get and apply a per-channel mask generated from np.random.binomial
        # 2) Scale your output accordingly
        # 3) During test time, you should not apply any mask or scaling.

        #zerp out entir channel

        retain_prob = 1 - self.p
        if train:
            sample = np.random.binomial(1, retain_prob, x.shape)

            x *= sample
            x /= retain_prob
        return x


    def backward(self, delta):
        """
        Arguments:
          delta (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
          np.array of same shape as input delta
        """
        # 1) This method is only called during training.


        return  delta * self.mask

