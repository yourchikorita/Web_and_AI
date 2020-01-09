import mlp
import activations as ac
import mnist
import numpy as np



np.random.seed(11785)

#initialize neural parameters
learning_rate = 0.004
momentum = 0.996 #0.956
num_bn_layers= 1
mini_batch_size = 10
epochs = 40

train, val, test = mnist.load_mnist()


net = mlp.MLP(784, 10, [64,32], [ac.Sigmoid(), ac.Sigmoid(), ac.Sigmoid()],
              ac.SoftmaxCrossEntropy(), learning_rate,
              momentum, num_bn_layers)

net.fit(train, val, epochs, mini_batch_size)


test_acc = net.validate(test) * 100.0
net.save(str(test_acc) + "_acc_nn_model.pkl")

print("Test Accuracy: " + str(test_acc) + "%")
