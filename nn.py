"""An implementation of a neural network without classes (just a module)
"""

import numpy
import scipy.optimize
import itertools

def create_training_dict(X, y):
    """Take a set of input features and their labels and package them
    along with some useful quantities into a dictionary.  This could
    be a training, validation, or test set.

    Args:
      X (numpy.ndarray): 2-D array of feature vectors (1 per row)
      y (numpy.ndarray): labels for each feature vector

    Returns:
      A dictionary containing ...
      X (numpy.ndarray): 2-D array of feature vectors (1 per row)
      y (numpy.ndarray): labels for each feature vector
      m (int): number of feature vectors (i.e. training examples)
      n (int): number of features per vector
      n_cat (int): number of categories (i.e. unique values in y)
      y1hot (numpy.ndarray) 2-D array of one-hot vectors (1 per row)
        for example if n_cat = 5, the label 3 -> [0, 0, 0, 1, 0]
    """
    m, n = X.shape
    n_cat = len(numpy.unique(y))
    y1hot = numpy.identity(n_cat)[y]
    Xmean = X.mean()
    Xstd = X.std()
    Xnorm = (X - Xmean) / Xstd
    return {'Xnorm': Xnorm, 'Xmean': Xmean, 'Xstd': Xstd, 'y': y, 'm': m,
            'n': n, 'n_cat': n_cat, 'y1hot': y1hot}


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2,s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


def sigmoid(z):
    """Return element-wise sigmoid

    Args:
      z (numpy.ndarray): argument for sigmoid function

    Returns:
      g (numpy.ndarray): sigmoid function evaluated element-wise
    """
    return 1.0 / (1.0 + numpy.exp(-z))


def sigmoid_gradient(z):
    """Return element-wise sigmoid gradient evaluated at z

    Args:
      z (numpy.ndarray): argument for sigmoid function

    Returns:
      g (numpy.ndarray): sigmoid function evaluated element-wise
    """
    return sigmoid(z) * (1.0 - sigmoid(z))


def flatten_arrays(arrays):
    """Turn a list of 2-D arrays into a single 1-D array.

    Args:
      arrays (``list`` of numpy.ndarray): a list of 2-D arrays

    Returns:
      (numpy.ndarray): a flattened 1-D array
    """
    return numpy.concatenate([a.flatten() for a in arrays])


def unflatten_array(flat_array, array_shapes):
    """Turn a single 1-D array into a list of 2-D arrays.

    Args:
      flat_array (numpy.ndarray): a flattened 1-D array
      array_shapes (``list`` of ``tuple``): 2-D array shapes

    Returns:
      arrays (``list`` of numpy.ndarray): a list of 2-D arrays
    """
    i = 0
    weight_arrays = []
    for shape in array_shapes:
        j = i + shape[0] * shape[1]
        weight_arrays.append(flat_array[i:j].reshape(shape))
        i = j
    return weight_arrays


def initialize_random_weights(layer_sizes):
    """Initialize weight arrays to random values.  We use the normalized
    initialization of Glorot and Bengio (2010).

    https://scholar.google.com/scholar?cluster=17889055433985220047&hl=en&as_sdt=0,22
    """
    weights = []
    for si, sj in pairwise(layer_sizes):
        b = numpy.sqrt(6.0 / (si + sj))
        weights.append(
            numpy.random.uniform(low=-b, high=b, size=(sj, si+1))
        )
    return weights


def minimize(initial_weights, X, y1hot, lam=0.0, method='TNC', jac=True,
             tol=1.0e-3, options={'disp': True, 'maxiter': 2000}):

    """Calculate values of weights that minimize the cost function.

    Args:
      initial_weights (``list`` of numpy.ndarray): weights between each layer
      X (numpy.ndarray): 2-D array of feature vectors (1 per row)
      y1hot (numpy.ndarray): 2-D array of one-hot vectors (1 per row)
      lam (``float``): regularization parameter
      method (``str``): minimization method (see scipy.optimize.minimize docs)
      jac (``bool`` or ``callable``): gradient provided? (see
        scipy.optimize.minimize docs)
      tol (``float``): stopping criterion (see scipy.optimize.minimize docs)
      options (``dict``): method specific (see scipy.optimize.minimize docs)

    Returns:
      res (``OptimizeResult``): (see scipy.optimize.minimize docs)
    """

    weight_shapes = [w.shape for w in initial_weights]
    flat_weights = flatten_arrays(initial_weights)
    res = scipy.optimize.minimize(
        compute_cost_and_grad,
        flat_weights,
        args=(X, y1hot, weight_shapes, lam),
        method=method,
        jac=jac,
        tol=tol,
        options=options,
    )
    return res


def compute_cost_and_grad(
        weights_flat, X, y1hot, weight_shapes, lam=0.0, cost_only=False):
    """Calculate cost function and its gradient with respect to weights.

    Args:
      weights_flat (numpy.ndarray): a flattened 1-D weight array
      X (numpy.ndarray): 2-D array of feature vectors (1 per row)
      y1hot (numpy.ndarray) 2-D array of one-hot vectors (1 per row)
      weight_shapes (``list`` of ``tuple``): 2-D array shapes
      lam (``float``): regularization parameter
      cost_only (``boolean``): if True return cost without gradient

    Returns:
      J (``float``): Cost with current weights
      weights_grad_flat (numpy.ndarray): d_J/d_weight
    """

    # package flat weights into a list of arrays
    m = X.shape[0]
    weights = unflatten_array(weights_flat, weight_shapes)

    # feed forward
    aa, zz = feed_forward(X, weights)

    # calculate raw cost
    h = aa[-1]
    J = -(
        numpy.sum(y1hot * numpy.log(h)) +
        numpy.sum((1.0 - y1hot) * numpy.log(1.0 - h))
    ) / m

    # add regularization
    for weight in weights:
        J += lam * numpy.sum(weight[:, 1:] * weight[:, 1:]) * 0.5 / m

    if cost_only:
        return J

    # gradient - back prop
    weights_grad_flat = flatten_arrays(
        back_propogation(weights, aa, zz, y1hot, lam=lam))

    return J, weights_grad_flat


def feed_forward(X, weights):
    """Perform a feed forward step.  Note that the z variables will
    not have the bias columns included and that all but the final a
    variables will have the bias column included.

    Args:
      X (numpy.ndarray): 2-D array of feature vectors (1 per row)
      weights (``list`` of numpy.ndarray): weights between each layer

    Returns:
      aa (``list`` of numpy.ndarray): activation of nodes for
        each layer.  The last item in the list is the hypothesis.
      zz (``list`` of numpy.ndarray): input into nodes for each layer.
    """
    aa = []
    zz = []

    zz.append(None) # this is z1 (i.e. there is no z1)
    ai = X.copy()
    ai = numpy.c_[numpy.ones(ai.shape[0]), ai] # a1 is X + bias nodes
    aa.append(ai)

    for weight in weights:
        zi = ai.dot(weight.T)
        zz.append(zi)
        ai = sigmoid(zi)
        ai = numpy.c_[numpy.ones(ai.shape[0]), ai] # add bias column
        aa.append(ai)

    # remove bias column from last aa layer
    aa[-1] = aa[-1][:, 1:]

    return aa, zz


def back_propogation(weights, aa, zz, y1hot, lam=0.0):
    """Perform a back propogation step

    Args:
      weights (``list`` of numpy.ndarray): weights between each layer
      aa (``list`` of numpy.ndarray): activation of nodes for
        each layer.  The last item in the list is the hypothesis.
      zz (``list`` of numpy.ndarray): input into nodes for each layer.
      y1hot (numpy.ndarray) 2-D array of one-hot vectors (1 per row)
      lam (``float``): regularization parameter

    Returns:
      weights_grad (``list`` of numpy.ndarray): d_J/d_weight
    """
    weights_grad = []
    m = y1hot.shape[0]
    n_layers = len(weights) + 1

    di_plus_1 = aa[-1] - y1hot
    i = n_layers - 2
    while i > 0:
        ones_col = numpy.ones(zz[i].shape[0])
        di = (
            di_plus_1.dot(weights[i]) *
            sigmoid_gradient(numpy.c_[ones_col, zz[i]])
        )
        di = di[:, 1:]
        weights_grad.append(di_plus_1.T.dot(aa[i]))
        i -= 1
        di_plus_1 = di.copy()

    weights_grad.append(di.T.dot(aa[0]))

    # we built it backwards
    weights_grad.reverse()

    # normalize by m
    weights_grad = [wg/m for wg in weights_grad]

    # add regularization (skip first columns)
    for i in range(n_layers-1):
        weights_grad[i][:, 1:] += lam/m * weights[i][:, 1:]

    return weights_grad
