import numpy as np
from loss import MSE


class CNN:

    def __init__(self, layers, loss_func=MSE):
        self.layers = layers
        self.params = []
        for layer in self.layers:
            self.params.append(layer[0].params)
        self.loss_func = loss_func

    def forward(self, X):
        for layer in self.layers:
            X = layer[0].conv_forward(X)
            print(X.shape)
            if layer[1]:
                X = layer[1].relu_forward(X)
        return X

    def backward(self, dout):
        grads = []
        for layer in reversed(self.layers):
            dout, grad = layer[0].conv_backward(dout)
            if layer[1]:
                dout = layer[1].relu_backward(dout)
            grads.append(grad)
        return grads

    def mse(self,A,B):
        mse_loss = (np.square(np.subtract(A, B))).mean()
        mse_grad = A - B
        return mse_loss, mse_grad

    def train_step(self, X, y):
        out = self.forward(X)
        loss, dloss = self.mse(out, y)

        grads = self.backward(dloss)

        return loss, grads
