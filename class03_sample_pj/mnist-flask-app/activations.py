import numpy as np

class Activation(object):
    """
    Interface for activation functions (non-linearities).
    """

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):

    def __init__(self):
        super(Sigmoid, self).__init__()
        self.state = None

    def forward(self, x):
        self.state = 1.0 / (1 + np.exp(-x))
        return self.state

    def derivative(self):
        return self.state * (1 - self.state)


class Tanh(Activation):

    def __init__(self):
        super(Tanh, self).__init__()
        self.state = None

    def forward(self, x):
        self.state = np.tanh(x)
        return self.state

    def derivative(self):
        return 1.0 - self.state ** 2


class ReLU(Activation):

    def __init__(self):
        super(ReLU, self).__init__()
        self.state = None
        self.x = None

    def forward(self, x):
        self.x = x
        self.state = x * (x > 0)
        return self.state

    def derivative(self):
        return np.where(self.state <= 0, 0, 1).astype(float)


class Criterion(object):
    """
    Interface for loss functions.
    """

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None

    def forward(self, x, y):
        self.logits = x - np.max(x)
        self.labels = y
        self.sm = (np.exp(self.logits).T / np.sum(np.exp(self.logits), axis=1)).T

        ce = -np.log(self.sm) * y
        loss = np.sum(ce, axis=1)
        return loss

    def derivative(self):
        grad = self.sm  - self.labels
        return grad