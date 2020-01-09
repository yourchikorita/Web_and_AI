import mlp
import numpy as np
import os
import glob

# Import image converter
from fig_to_mnist import mnist_treat

def classify(fig):

    # Standard MNIST format is 28x28 pixels and grayscale.
    height, width, depth = 28, 28, 1

    # Proccess image (c.f. fig_to_misc.py) and convert to numpy array
    X = mnist_treat(fig)
    X_array = np.array(X)

    # Flatten image
    X_flat = X_array.reshape(1, height * width)

    # Normalize to 1 = white.
    X_flat = X_flat.astype('float32')
    X_flat /= 255

    net = mlp.MLP()

    # load latest model
    path = os.path.join(os.curdir, 'models/*')
    files = sorted(
        glob.iglob(path), key=os.path.getctime, reverse=True)

    net.load(files[0])

    # Run prediction
    prediction = net.predict(X_flat)


    return prediction[0]
