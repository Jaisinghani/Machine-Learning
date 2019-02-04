import numpy as np


class Convolution():

    def __init__(self, input_channels, output_channels, filter_size, padding):
        self.weights = self.init_weights((filter_size, filter_size), input_channels, output_channels)
        self.bias = self.init_bias(output_channels)
        self.padding = padding
        self.params=(self.weights,self.bias)
        self.A_prev = None
        self.A_curr=None

    def init_weights(self, filter_size, ip_channels, op_channels):
        filter_height, filter_width = filter_size
        # a=np.random.randn(filter_height, filter_width, ip_layer, op_layer)
        # print("std:",np.std(a),np.sqrt(
        #     2 / (ip_layer*filter_width*filter_height)))
        w = np.random.randn(filter_height, filter_width, ip_channels, op_channels) * np.sqrt(
            2 / (ip_channels))
        # w = np.random.normal(loc=0.0, scale=0.1, size=(filter_height, filter_width, ip_layer, op_layer)) * np.sqrt(1 / 1000000)
        return w

    def init_bias(self, op_channels):
        return np.zeros((1, 1, 1, op_channels))

    def zero_pad(self, inp, pad):
        X_pad = np.pad(inp, ((pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
        return X_pad

    def conv_single_step(self, a_slice_prev, W, b):
        s = np.multiply(a_slice_prev, W) + b
        Z = np.sum(s)
        return Z

    def conv_forward(self, A_prev):
        self.A_prev = A_prev
        (n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        (f, f, n_C_prev, n_C) = self.weights.shape
        pad = self.padding
        n_H = int((n_H_prev - f + 2 * pad)) + 1
        n_W = int((n_W_prev - f + 2 * pad)) + 1
        Z = np.zeros((n_H, n_W, n_C))
        A_prev_pad = self.zero_pad(A_prev, pad)
        for h in range(n_H):  # loop over vertical axis of the output volume
            for w in range(n_W):  # loop over horizontal axis of the output volume
                for c in range(n_C):  # loop over channels (= #filters) of the output volume
                    a_slice_prev = A_prev_pad[h:h + f, w:w + f, :]
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                    Z[h, w, c] = self.conv_single_step(a_slice_prev, self.weights[..., c], self.bias[..., c])

        assert (Z.shape == (n_H, n_W, n_C))

        # Save information in "cache" for the backprop
        cache = (A_prev, self.weights, self.bias, self.padding)
        self.A_curr=Z
        return Z

    def conv_backward(self, dZ):
        # (A_prev, W, b, hparameters) = cache

        (n_H_prev, n_W_prev, n_C_prev) = self.A_prev.shape

        (f, f, n_C_prev, n_C) = self.weights.shape

        pad = self.padding

        (n_H, n_W, n_C) = dZ.shape

        dA_prev = np.zeros((n_H_prev, n_W_prev, n_C_prev))
        dW = np.zeros((f, f, n_C_prev, n_C))
        db = np.zeros((1, 1, 1, n_C))

        A_prev_pad = self.zero_pad(self.A_prev, pad)
        dA_prev_pad = self.zero_pad(dA_prev, pad)

        for h in range(n_H):  # loop over vertical axis of the output volume
            for w in range(n_W):  # loop over horizontal axis of the output volume
                for c in range(n_C):  # loop over the channels of the output volume

                    a_slice = A_prev_pad[h:h + f, w:w + f, :]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    dA_prev_pad[h:h + f, w:w + f, :] += self.weights[:, :, :, c] * dZ[h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[h, w, c]
                    db[:, :, :, c] += dZ[h, w, c]

        dA_prev[:, :, :] = dA_prev_pad[pad:-pad, pad:-pad, :]

        # Making sure your output shape is correct
        assert (dA_prev.shape == (n_H_prev, n_W_prev, n_C_prev))

        return dA_prev, (dW, db)
