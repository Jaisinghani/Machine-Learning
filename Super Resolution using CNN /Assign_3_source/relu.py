import numpy as np


class ReLU():
    def __init__(self):
        self.X = None

    def relu_forward(self, X):
        """
        Computes the forward pass for a layer of rectified linear units (ReLUs).
        Input:
        - x: Inputs, of any shape
        """
        self.X = X
        self.X[self.X <= 0] = 0
        return self.X

    def relu_backward(self, dout):
        """
         Computes the backward pass for a layer of rectified linear units (ReLUs).
         Input:
        """
        dX = dout.copy()
        dX[self.X <= 0] = 0
        return dX
