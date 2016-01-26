"""Neural Network Python Implementation."""
import scipy.io
import numpy
import nn
import random
import matplotlib.pyplot as plt

# load training data
# this is m=50000 images of n=28x28=784 pixels each
# each image is a digit in the range 0-9
#====================================================================
mnist_data = numpy.load('data/mnist_uint8_uint8.npz')
X = mnist_data['train_features']
y = mnist_data['train_labels']
m, n = X.shape
n_labels = len(numpy.unique(y))

# create one-hot vectors
#====================================================================

# the identity matrix contains all one_hot vectors we need
Iy = numpy.identity(n_labels)

# make a matrix in which each row is a one_hot vector
yarr = Iy[y]

# hard code layer sizes and initialize random weights
#====================================================================
n_input_nodes = 784
n_hidden_nodes = 65
n_output_nodes = 10
layer_sizes = [n_input_nodes, n_hidden_nodes, n_output_nodes]


# initialize random weights and train
#====================================================================
random_weights = nn.initialize_random_weights(layer_sizes)
weight_shapes = [w.shape for w in random_weights]

lam = 1.0
res = nn.minimize(random_weights, X, yarr, weight_shapes, lam=lam)
trained_weights = nn.unflatten_array(res.x, weight_shapes)

# make predictions
Xtest = mnist_data['test_features']
ytest = mnist_data['test_labels']
aa, zz = nn.feed_forward(Xtest, trained_weights)
h = aa[-1]
y_predict = numpy.argmax(h, axis=1)

accuracy = (ytest==y_predict).astype(numpy.float).mean()
print
print 'Test Set Accuracy (Neural Network): ', accuracy
print

plt.ion()
for ishow in range(10):
    i = random.randint(0, ytest.size-1)
    print 'using image {}'.format(i)
    print 'actual: {}, prediction: {}'.format(ytest[i], y_predict[i])
    plt.imshow(Xtest[i,:].reshape(28,28))
    go_again = raw_input('press enter to continue ... ')
plt.ioff()
