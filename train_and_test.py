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


# make learning curve
#====================================================================
lam = 0.0
m_samples = [(i+1)*1000 for i in range(50)]

train_accs = []
valid_accs = []

train_Js = []
valid_Js = []

for im in m_samples:

    # train
    res = nn.minimize(
        random_weights,
        train['Xnorm'][0:im,:],
        train['y1hot'][0:im,:],
        lam=lam)

    trained_weights_flat = res.x
    trained_weights = nn.unflatten_array(res.x, weight_shapes)

    # compute cost of training sample
    J_train = nn.compute_cost_and_grad(
        trained_weights_flat,
        train['Xnorm'][0:im,:],
        train['y1hot'][0:im,:],
        weight_shapes,
        lam=lam,
        cost_only=True)
    train_Js.append(J_train)

    # compute accuracy on validation sample
    aa, zz = nn.feed_forward(train['Xnorm'][0:im,:], trained_weights)
    h = aa[-1]
    y_predict = numpy.argmax(h, axis=1)
    accuracy = (train['y'][0:im] == y_predict).astype(numpy.float).mean()
    train_accs.append(accuracy)


    # compute cost on validation sample
    J_valid = nn.compute_cost_and_grad(
        trained_weights_flat,
        valid['Xnorm'],
        valid['y1hot'],
        weight_shapes,
        lam=lam,
        cost_only=True)
    valid_Js.append(J_valid)

    # compute accuracy on validation sample
    aa, zz = nn.feed_forward(valid['Xnorm'], trained_weights)
    h = aa[-1]
    y_predict = numpy.argmax(h, axis=1)
    accuracy = (valid['y'] == y_predict).astype(numpy.float).mean()
    valid_accs.append(accuracy)

    print
    print 'Validation, Training Set Cost (m={}): {}, {}'.format(
        im, valid_Js[-1], train_Js[-1])
    print 'Validation, Training Set Accuracy (m={}): {}, {}'.format(
        im, valid_accs[-1], train_accs[-1])
    print


plt.plot(train_Js, lw=3.0, ls='--', color='blue', label='Jtrain')
plt.plot(valid_Js, lw=3.0, ls='-', color='red', label='Jcv')
plt.legend(loc='best')

sys.exit(1)


lamvals = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
accuracies = []
ypredicts = []
for lam in lamvals:

    # train
    res = nn.minimize(random_weights, Xtrain, y1hot, weight_shapes, lam=lam)
    trained_weights = nn.unflatten_array(res.x, weight_shapes)

    # accuracy for validation set
    aa, zz = nn.feed_forward(Xvalid, trained_weights)
    h = aa[-1]
    ypredict = numpy.argmax(h, axis=1)
    accuracy = (yvalid == ypredict).astype(numpy.float).mean()

    ypredicts.append(ypredict)
    accuracies.append(accuracy)

    print
    print 'Validation Set Accuracy (lambda={}): {}'.format(lam, accuracy)
    print

sys.exit(1)






plt.ion()
for ishow in range(10):
    i = random.randint(0, ytest.size-1)
    print 'using image {}'.format(i)
    print 'actual: {}, prediction: {}'.format(ytest[i], y_predict[i])
    plt.imshow(Xtest[i,:].reshape(28,28))
    go_again = raw_input('press enter to continue ... ')
plt.ioff()
