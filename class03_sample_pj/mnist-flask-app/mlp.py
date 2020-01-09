"""

"""

import numpy as np
import os
import pickle
import time

import batchnorm as bn
import activations as ac

np.random.seed(11785)


class MLP(object):
    """
    A multilayer perceptron with batch normalization and momentum
    """


    def __init__(self, input_size=784, output_size=10, hiddens=[64,32], activations=[ac.Sigmoid(), ac.Sigmoid(), ac.Sigmoid()], \
                  criterion=ac.SoftmaxCrossEntropy(),\
                 lr=0.008, momentum=0.856, num_bn_layers=1):
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum

        self.W = None
        self.b = None

       # self.weight_init_fn = weight_init_fn
       # self.bias_init_fn = bias_init_fn

        if self.bn:
            self.bn_layers = [bn.BatchNorm(hiddens[t]) for t in range(0,num_bn_layers)]

        self.loss = None
        self.firstinit = True

        self.hiddens = hiddens

        self.zerosW = None
        self.zerosb = None

        self.batch_size = 10
        self.epochs = 40

        self.training_loss = []
        self.validation_acc = []

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

    # random weight init
    def weight_init(self, x, y):
        return np.random.randn(x, y)

    # zero bias init
    def bias_init(self, x):
        return np.zeros((1, x))


    def re_init(self):

        """ Model init function.
        """

        self.firstinit = False

        layer_size = self.input_size
        weights = []
        biases = []

        #layers
        if len(self.hiddens):
            for idx in range(self.nlayers-1):
                weights.append(self.weight_init(layer_size,self.hiddens[idx]))
                layer_size = self.hiddens[idx]

                biases.append(self.bias_init(layer_size))

        # output layer
        weights.append(self.weight_init(layer_size,self.output_size))
        biases.append(self.bias_init(self.output_size))

        self.W = weights
        self.b = biases

        self.pW = [np.zeros(t.shape) for t in self.W]
        self.pb = [np.zeros(t.shape) for t in self.b]


        self.dW = [np.zeros(t.shape) for t in self.W]
        self.db = [np.zeros(t.shape) for t in self.b]


        self.zerosW = [np.zeros(t.shape) for t in self.W]
        self.zerosb = [np.zeros(t.shape) for t in self.b]

        self.training_loss = []
        self.validation_acc = []

    def zero_grads(self):
        """ Gradient zeroing.
        """
        self.dW = self.zerosW
        self.db = self.zerosb

    def step(self):
        """Weights update step function.
        """
        for idx in reversed(range(self.nlayers)):
            self.pW[idx] = self.pW[idx] * self.momentum - self.lr*self.dW[idx]
            self.pb[idx] = self.pb[idx] * self.momentum - self.lr*self.db[idx]
            self.W[idx] += self.pW[idx]
            self.b[idx] += self.pb[idx]

        if self.bn:
            for idx,bn in enumerate(self.bn_layers):
                bn.gamma -= bn.dgamma*self.lr
                bn.beta -= bn.dbeta * self.lr

    def forward(self, x):
        """Forward propagation.

        Parameters
        ----------
        x : input


        Returns
        -------
        array
            Last layer activations result.

        """
        self.input = x
        self.logits = []

        if self.firstinit and self.train_mode: self.re_init()

        # input layer
        step = self.input
        self.logits.append(step)

        for idx in range(self.nlayers):
            step = np.dot(step,self.W[idx]) + self.b[idx]

            if idx < self.num_bn_layers:
                step = self.bn_layers[idx](step, self.train_mode)

            step = self.activations[idx](step)
            self.logits.append(step)

        return step


    def backward(self, labels):
        """Backward gradient descent.

        Parameters
        ----------
        labels : list

        """
        self.loss = self.criterion.forward(self.logits[-1], labels)
        self.batch_size = self.input.shape[0]
        grad_loss = (1 / self.batch_size) * self.criterion.derivative()

        dinput = grad_loss

        for idx in reversed(range(self.nlayers)):
            grad_loss = self.activations[idx].derivative() * dinput

            if idx < self.num_bn_layers:
                grad_loss = self.bn_layers[idx].backward(grad_loss)

            self.dW[idx] = np.dot(self.logits[idx].T, grad_loss)
            self.db[idx] = np.sum(grad_loss, axis=0, keepdims=False)
            dinput = np.dot(grad_loss, self.W[idx].T)


    def fit(self, training_data, validation_data=None, nepochs=40, batch_size=10):
        """Fit (train) the MLP on provided training data.

        Parameters
        ----------
        training_data : array of lists
            [0],[1] = image, label.

        validation_data : array of lists, optional
            If provided, the network will count
            validation accuracy after each epoch.

        nepochs : number of epochs, optional
            By default it equals 40

        batch_size : size of minibatches, optional
            By default it equals 10

        """
        self.train()
        #training_losses = [] TODO
        #training_errors = []
        #validation_accuracy = []
        #validation_errors = []

        self.epochs = nepochs
        self.batch_size = batch_size
        self.firstinit = True
        self.re_init()

        # np.random.shuffle(training_data)
        x1, y1 = training_data

        mini_batches_x = [
            x1[k:k + self.batch_size] for k in
            range(0, len(x1), self.batch_size)]

        mini_batches_y = [
            y1[k:k + self.batch_size] for k in
            range(0, len(y1), self.batch_size)]

        for epoch in range(self.epochs):

            for b in range(0, len(mini_batches_x)):
                x = mini_batches_x[b]
                y = mini_batches_y[b]

                self.forward(x)
                self.backward(y)
                self.step()

            self.training_loss.append(np.mean(self.loss))

            if validation_data:
                accuracy = self.validate(validation_data) * 100.0
                print("Epoch {0}, accuracy {1} %.".format(epoch + 1, accuracy))
                self.validation_acc.append(accuracy)
            else:
                print("Processed epoch {0}.".format(epoch))


    def validate(self, validation_data):
        """Function uses the
        number of correctly predicted classes as validation accuracy metric.

        Parameters
        ----------
        validation_data : list

        Returns
        -------
        int
            Percent of correctly predicted classes.
        """
        counter = 0
        for idx, x in enumerate(validation_data[0]):
            if self.predict(x) == validation_data[1][idx]:
                counter += 1

        return counter/len(validation_data[1])

    def predict(self, x):
        """Predict the class of a single test example.

        Parameters
        ----------
        x : numpy.array

        Returns
        -------
        int
            Predicted class.

        """
        self.eval()
        self.forward(x)
        predicted = np.argmax(self.logits[-1], axis=1)
        return predicted

    def load(self, filename='nn_model.pkl'):
        """Load serialized model with weights and biases

        Parameters
        ----------
        filename : str, optional
        Name of the ``.pkl`` serialized object.

        """
        with open(filename,'rb') as f:
            nn_model = pickle.load(f, encoding='bytes')
        f.close()

        self.W = nn_model.W
        self.b = nn_model.b

        self.num_bn_layers = nn_model.num_bn_layers
        self.bn = nn_model.num_bn_layers > 0
        self.hiddens = nn_model.hiddens
        self.nlayers = len(nn_model.hiddens) + 1
        self.input_size = nn_model.input_size
        self.output_size = nn_model.output_size
        self.activations = nn_model.activations
        self.criterion = nn_model.criterion
        self.lr = nn_model.lr
        self.momentum = nn_model.momentum

        if self.bn:
            self.bn_layers = nn_model.bn_layers

        self.train_mode = nn_model.train_mode
        self.firstinit= nn_model.firstinit
        self.zerosW = nn_model.zerosW
        self.zerosb = nn_model.zerosb

        self.batch_size = nn_model.batch_size
        self.epochs = nn_model.epochs

    def save(self, filename='nn_model.pkl'):
        """Save serialized model of neural network

        Parameters
        ----------
        filename : str, optional
        Name of the ``.pkl`` serialized object

        """
        seconds = time.time()

        directory = os.path.join(os.curdir, 'models')
        filepath = os.path.join(directory, str(seconds)+'_'+filename)

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        f.close()