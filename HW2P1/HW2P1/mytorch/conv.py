# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        self.x = x

        batch_size, in_channel, input_size = x.shape

        output_size = ((input_size - self.kernel_size)//self.stride) + 1
        output = np.zeros([batch_size, self.out_channel, output_size])

        for b in range(batch_size):
            for out_c in range(self.out_channel):
                for w in range( output_size):
                    segment = x[b,:, (w*self.stride):(w*self.stride + self.kernel_size)]
                    output[b, out_c,  w] = np.sum(segment*self.W[out_c])
                output[b,out_c] =  output[b,out_c] + self.b[out_c]
        return output

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        batch_size, out_channel, output_size = delta.shape

        in_channel = self.x.shape[1]
        input_size = self.x.shape[2]

        dx = np.zeros([batch_size, in_channel, input_size])
        for b in range(batch_size):
            for in_c in range(in_channel):
                for w in range(output_size):

                    lb = w*self.stride
                    ub = w*self.stride + self.kernel_size
                    for k in range(lb, ub):
                        dx[b,in_c,k] += sum(self.W[out_c, in_c, (k-w*self.stride)] * delta[b,out_c, w] for out_c in range(out_channel))

        self.dW = np.zeros(self.W.shape)

        for b in range(batch_size):
            for k in range(self.kernel_size):
                for in_c in range(in_channel):
                    for out_c in range(out_channel):
                        self.dW[out_c, in_c, k] += sum([self.x[b, in_c, w* self.stride+k] * delta[b, out_c, w] for w in range(output_size)])


        self.db = np.sum(delta,axis = (0,2))
        return dx


class Conv2D():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)


        """
        """
        self.x = x
        batch_size = x.shape[0]
        in_channel = x.shape[1]
        width = x.shape[2]
        height = x.shape[3]


        output_size = ((width - self.kernel_size)//self.stride) + 1

        output = np.zeros([batch_size, self.out_channel, output_size, output_size])


        for b in range(0, batch_size):
            for out in range(0,self.out_channel):

                for m in range(0, output_size): #refers to vertically
                    for n in range(0, output_size): # refers to horizontally


                        low = n * self.stride

                        low2 = m * self.stride

                        output[b,out,m,n] = np.sum( x[b, : ,low2 :low2+self.kernel_size, low:low+self.kernel_size] * self.W[out])
            output[b,out] +=   self.b[out]

        return output"""

        self.x = x

        batch_size, in_channel, input_w, input_h = x.shape
        output_w = int(((input_w - self.kernel_size)//self.stride) + 1)
        output_h = int(((input_h - self.kernel_size)//self.stride) + 1)
        output = np.zeros([batch_size, self.out_channel, output_w, output_h])

        for b in range(batch_size):
            for out_c in range(self.out_channel):
                for w in range(output_w):
                    for h in range(output_h):
                        segment = x[b,:, (w*self.stride):(w*self.stride + self.kernel_size), (h*self.stride):(h*self.stride + self.kernel_size)]
                        output[b, out_c,  w, h] = np.sum(np.multiply(segment, self.W[out_c]))
                output[b,out_c] = output[b,out_c] + self.b[out_c]
        return output

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        batch_size, in_channel, input_w, input_h = self.x.shape
        output_w = int(((input_w - self.kernel_size)//self.stride) + 1)

        output_h = int(((input_h - self.kernel_size)//self.stride) + 1)
        dx = np.zeros([batch_size, self.in_channel, input_w, input_h])
        for b in range(batch_size):
            for in_c in range(in_channel):
                for w in range(output_w):
                    for h in range(output_h):
                        lbw = w*self.stride
                        ubw = w*self.stride + self.kernel_size
                        lbh = h*self.stride
                        ubh = h*self.stride + self.kernel_size
                        for nw in range(lbw, ubw):
                            for nh in range(lbh, ubh):
                                dx[b, in_c,nw, nh]  += sum([self.W[out_c, in_c, nw-lbw, nh-lbh] * delta[b, out_c, w, h] for out_c in range(self.out_channel)])

        self.db = np.sum(np.sum(delta,axis = (0,2)), axis = 1)

        for out_c in range(self.out_channel):
            for k in range(self.kernel_size):
                for s in range(self. kernel_size):
                    for in_c in range(in_channel):
                        for b in range(batch_size):
                            for w in range(output_w):
                                for h in range(output_h):
                                    lbw = w*self.stride
                                    #ubw = w*self.stride + self.kernel_size
                                    lbh = h*self.stride
                                    #ubh = h*self.stride + self.kernel_size

                                    self.dW[out_c, in_c, k, s] += self.x[b,in_c,lbw+k, lbh+s] * delta[b,out_c, w, h]

        return dx

class Conv2D_dilation():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride, padding=0, dilation=1,
                 weight_init_fn=None, bias_init_fn=None):
        """
        Much like Conv2D, but take two attributes into consideration: padding and dilation.
        Make sure you have read the relative part in writeup and understand what we need to do here.
        HINT: the only difference are the padded input and dilated kernel.
        """

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # After doing the dilation， the kernel size will be: (refer to writeup if you don't know)
        self.kernel_dilated = (self.kernel_size - 1) * (self.dilation - 1) + self.kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        self.W_dilated = np.zeros((self.out_channel, self.in_channel, self.kernel_dilated, self.kernel_dilated))

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        # TODO: padding x with self.padding parameter (HINT: use np.pad())

        # TODO: do dilation -> first upsample the W -> computation: k_new = (k-1) * (dilation-1) + k = (k-1) * d + 1
        #       HINT: for loop to get self.W_dilated

        # TODO: regular forward, just like Conv2d().forward()

        # general order: pad -> dilation -> call Conv2D().forward()
        batch_size, in_channel, input_w, input_h = x.shape
        self.pad_x = np.zeros(x.shape)
        for b in range(batch_size):
            for in_c in range(in_channel):
                self.pad_x[b, in_c] = np.pad(x[b, in_c, :, :], self.padding, mode='constant')

        # dilated the W, add number into the W_dilated,

        for i in range(self.kernel_dilated):
            for out_c in range(self.out_channel):
                for in_c in range(self.in_channel):
                    if i == 1:
                        self.W_dilated[out_c, in_c, i, i] = self.W_dilated[out_c, in_c, i, i]
                    else:
                        self.W_dilated[out_c, in_c, i + self.stride, i + self.stride] = self.W_dilated[out_c, in_c, i, i]


        batch_size, in_channel, input_w, input_h = self.pad_x.shape
        output_w = int(((input_w - self.kernel_dilated)//self.stride) + 1)
        output_h = int(((input_h - self.kernel_dilated)//self.stride) + 1)
        output = np.zeros([batch_size, self.out_channel, output_w, output_h])

        for b in range(batch_size):
            for out_c in range(self.out_channel):
                for w in range(output_w):
                    for h in range(output_h):
                        segment = x[b,:, (w*self.stride):(w*self.stride + self.kernel_dilated), (h*self.stride):(h*self.stride + self.kernel_dilated)]
                        output[b, out_c,  w, h] = np.sum(np.multiply(segment, self.W[out_c]))
                output[b,out_c] = output[b,out_c] + self.b[out_c]
        return output

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        # TODO: main part is like Conv2d().backward(). The only difference are: we get padded input and dilated kernel
        #       for whole process while we only need original part of input and kernel for backpropagation.
        #       Please refer to writeup for more details.

        batch_size, in_channel, input_w, input_h = self.pad_x.shape
        output_w = int(((input_w - sself.kernel_dilated)//self.stride) + 1)

        output_h = int(((input_h - self.kernel_dilated)//self.stride) + 1)
        dx = np.zeros([batch_size, self.in_channel, input_w, input_h])
        for b in range(batch_size):
            for in_c in range(in_channel):
                for w in range(output_w):
                    for h in range(output_h):
                        lbw = w*self.stride
                        ubw = w*self.stride + self.kernel_dilated
                        lbh = h*self.stride
                        ubh = h*self.stride + self.kernel_dilated
                        for nw in range(lbw, ubw):
                            for nh in range(lbh, ubh):
                                dx[b, in_c,nw, nh]  += sum([self.W_dilated[out_c, in_c, nw-lbw, nh-lbh] * delta[b, out_c, w, h] for out_c in range(self.out_channel)])

        self.db = np.sum(np.sum(delta,axis = (0,2)), axis = 1)

        for out_c in range(self.out_channel):
            for k in range(self.kernel_dilated):
                for s in range(self.kernel_dilated):
                    for in_c in range(in_channel):
                        for b in range(batch_size):
                            for w in range(output_w):
                                for h in range(output_h):
                                    lbw = w*self.stride
                                    #ubw = w*self.stride + self.kernel_size
                                    lbh = h*self.stride
                                    #ubh = h*self.stride + self.kernel_size

                                    self.dW[out_c, in_c, k, s] += self.pad_x[b,in_c,lbw+k, lbh+s] * delta[b,out_c, w, h]

        return dx


class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        return x.reshape(self.b, self.c * self.w)

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        dx = np.reshape(delta, (self.b, self.c, self.w))
        return dx

