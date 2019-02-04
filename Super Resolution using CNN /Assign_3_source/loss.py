import numpy as np


class MSE:
    def mse(self, A, B):
        mse_loss = (np.square(np.subtract(A, B))).mean()
        mse_grad = A - B
        return mse_loss, mse_grad
