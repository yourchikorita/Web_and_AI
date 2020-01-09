import numpy as np


class BatchNorm(object):

    def __init__(self, fan_in, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        self.x = x

        if eval: #eval mode
            self.mean = np.mean(self.x, axis=0)
            self.var = np.var(self.x, axis=0)

            self.norm = (self.x - self.mean) / np.sqrt(self.var + self.eps)
            self.out = self.gamma * self.norm  + self.beta

            self.running_mean = self.running_mean * self.alpha + self.mean * (1 - self.alpha)
            self.running_var = self.running_var * self.alpha + self.var * (1 - self.alpha)

        else: # running mode
            x_normed = (self.x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            self.out = self.gamma * x_normed + self.beta

        return self.out


    def backward(self, delta):

        # https://github.com/kevinzakka/research-paper-notes/blob/master/batch_norm.py

        n, d = delta.shape

        x_mu = self.x - self.mean
        inv_var = 1. / np.sqrt(self.var + self.eps)

        x_hat = self.norm

        # intermediate partial derivatives
        dxhat = delta * self.gamma
        dvar = np.sum((dxhat * x_mu * (-0.5) * (inv_var) ** 3), axis=0)
        dmu = (np.sum((dxhat * -inv_var), axis=0)) + (dvar * (-2.0 / n) * np.sum(x_mu, axis=0))
        dx1 = dxhat * inv_var
        dx2 = dvar * (2.0 / n) * x_mu
        dx3 = (1.0 / n) * dmu

        # final partial derivatives
        dx = dx1 + dx2 + dx3

        dout = n * delta

        self.dbeta = np.sum(dout, axis=0)
        self.dgamma = np.sum(x_hat * dout, axis=0)

        return dx