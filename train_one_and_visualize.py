"""Neural Network Python Implementation."""
import numpy
import random
import matplotlib.pyplot as plt
import nn

# load training, validatin, and test data
# train: m=50000 images of n=28x28=784 pixels each
# valid: m=10000 images of n=28x28=784 pixels each
# test: m=10000 images of n=28x28=784 pixels each
# each image (and label) is a digit in the range 0-9
#====================================================================
mnist_data = numpy.load('data/mnist_uint8_uint8.npz')

train = nn.create_training_dict(
    mnist_data['train_features'], mnist_data['train_labels'])

valid = nn.create_training_dict(
    mnist_data['valid_features'], mnist_data['valid_labels'])

test = nn.create_training_dict(
    mnist_data['test_features'], mnist_data['test_labels'])


# hard code layer sizes and initialize random weights
#====================================================================
n_input_nodes = 784  # pixels in image
n_hidden_nodes = 25  # variable
n_output_nodes = 10  # number of labels
layer_sizes = [n_input_nodes, n_hidden_nodes, n_output_nodes]


# set initial random weights
#====================================================================
random_weights = nn.initialize_random_weights(layer_sizes)
weight_shapes = [w.shape for w in random_weights]


# train
#====================================================================
lam = 3.0

train_accs = []
valid_accs = []

train_Js = []
valid_Js = []

# train
res = nn.minimize(random_weights, train['Xnorm'], train['y1hot'], lam=lam)

trained_weights_flat = res.x
trained_weights = nn.unflatten_array(res.x, weight_shapes)

# compute cost of training sample
J_train = nn.compute_cost_and_grad(
    trained_weights_flat, train['Xnorm'], train['y1hot'], weight_shapes,
    lam=lam, cost_only=True)

# compute accuracy on training sample
aa, zz = nn.feed_forward(train['Xnorm'], trained_weights)
h = aa[-1]
y_predict = numpy.argmax(h, axis=1)
train_accuracy = (train['y'] == y_predict).astype(numpy.float).mean()

# compute cost on validation sample
J_valid = nn.compute_cost_and_grad(
    trained_weights_flat, valid['Xnorm'], valid['y1hot'], weight_shapes,
    lam=lam, cost_only=True)

# compute accuracy on validation sample
aa, zz = nn.feed_forward(valid['Xnorm'], trained_weights)
h = aa[-1]
y_predict = numpy.argmax(h, axis=1)
valid_accuracy = (valid['y'] == y_predict).astype(numpy.float).mean()


print
print 'Validation, Training Set Cost (m={}): {}, {}'.format(
    train['m'], J_valid, J_train)
print 'Validation, Training Set Accuracy (m={}): {}, {}'.format(
    train['m'], valid_accuracy, train_accuracy)
print


# plot some random results
#====================================================================

aa, zz = nn.feed_forward(valid['Xnorm'], trained_weights)
h = aa[-1]
y_predict = numpy.argmax(h, axis=1)

plt.ion()
for ishow in range(10):
    i = random.randint(0, valid['m'] - 1)
    print 'using image {}'.format(i)
    print 'actual: {}, prediction: {}'.format(valid['y'][i], y_predict[i])
    plt.imshow(valid['Xnorm'][i,:].reshape(28,28))
    go_again = raw_input('press enter to continue ... ')
plt.ioff()
