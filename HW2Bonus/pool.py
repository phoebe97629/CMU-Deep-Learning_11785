import numpy as np


class MaxPoolLayer():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride


    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """



        self.x = x

        batch_size, in_channel, input_width, input_height = self.x.shape
        output_width = (input_width - self.kernel) // self.stride + 1
        output_height = (input_height - self.kernel) // self.stride + 1

        out = np.zeros([batch_size, in_channel, output_width, output_height])
        self.arg_max = np.zeros((batch_size, in_channel, output_width, output_height), dtype=int)

        for b in range(batch_size):
            for out_c in range(in_channel):
                for w in range(output_width):
                    for h in range(output_height):
                        segment = x[b, out_c, w*self.stride: self.stride * w + self.kernel,
                                            self.stride * h: self.stride * h + self.kernel]
                        out[b, out_c, w, h] = np.max(segment)
                        self.arg_max[b, out_c, w, h] = np.argmax(segment)


        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        batch_size,in_channel, out_width, out_height = delta.shape
        _, _, in_width, in_height = self.x.shape
        dx = np.zeros((batch_size, in_channel, in_width, in_height))
        for b in range(batch_size):
            for c in range(in_channel):
                for w in range(out_width):
                    for h in range(out_height):
                        #find the max index

                        idx = np.unravel_index(self.arg_max[b, c, w, h], (self.kernel, self.kernel))
                        w_idx = self.stride * w + idx[0]
                        h_idx = self.stride * h + idx[1]
                        dx[b, c, w_idx, h_idx] = delta[b, c, w, h]

        return dx


class MeanPoolLayer():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        self.x = x
        batch_size, in_channel, input_width, input_height = x.shape
        output_width = (input_width - self.kernel) // self.stride + 1
        output_height = (input_height - self.kernel) // self.stride + 1

        out = np.zeros([batch_size, in_channel, output_width, output_height])

        for b in range(batch_size):
            for out_c in range(in_channel):
                for w in range(output_width):
                    for h in range(output_height):
                        segment = x[b, out_c, (w * self.stride):(w * self.stride + self.kernel),
                                  (h * self.stride):(h * self.stride + self.kernel)]
                        out[b, out_c, w, h] = np.mean(segment)
        return out
        # raise NotImplementedError

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        dx = np.zeros_like(self.x)
        batch_size, in_channel, input_width, input_height = self.x.shape
        output_width = (input_width - self.kernel) // self.stride + 1
        output_height = (input_height - self.kernel) // self.stride + 1

        for b in range(batch_size):
            for c in range(in_channel):
                for w in range(output_width):
                    for h in range(output_height):
                        dx[b, c, (w * self.stride):(w * self.stride + self.kernel),
                        (h * self.stride):(h * self.stride + self.kernel)] += delta[b,c, w,h] / (self.kernel**2)

        return dx
