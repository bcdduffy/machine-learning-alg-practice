"""
Functions for training and predicting with linear classifiers
"""
import numpy as np
import matplotlib.pylab as plt
#import pylab as plt
from scipy.optimize import minimize, check_grad
def linear_predict(data, model):
    """
    Predicts a multi-class output based on scores from linear combinations of
features.
    :param data: size (n, m) ndarray containing m examples described by n features
each
    :type data: ndarray
    :param model: dictionary containing 'weights' key. The value for the 'weights'
key is a size
        (n, num_classes) ndarray
    :type model: dict
    :return: length m vector of class predictions
    :rtype: array
    """

    model_weights = model["weights"]
    trans_weights = np.transpose(model_weights)
    #print(trans_weights)
    #print(data)
    linear_sum = np.dot(trans_weights,data)
    #print("lin sum")
    #print(linear_sum)
    linear_comb = linear_sum.argmax(0)
    ##print("lin comb")
    #print(linear_comb.shape())
    return linear_comb

def perceptron_update(data, model, params, label):
    """
    Update the model based on the perceptron update rule and return whether the
perceptron was correct
    :param data: (n, 1) ndarray representing one example input
    :type data: ndarray
    :param model: dictionary containing 'weights' key. The value for the 'weights'
key is a size
        (n, num_classes) ndarray
    :type model: dict
    :param params: dictionary containing 'alpha' key. alpha is the learning rate
and it should be a float
    :type params: dict
    :param label: the class label of the single example
    :type label: int
    :return: whether the perceptron correctly predicted the provided true label of
the example
    :rtype: bool
    """

# and returning the proper boolean value
#print(model["weights"])
    prediciton = linear_predict(data, model)
    alpha = params["alpha"]
    alpha = np.ones((2,4))
    pred_arr = np.ones(model["weights"].shape)*prediciton*alpha
    class_ind = np.arange(0,4).reshape(1,4)
    #class_min_pred = np.subtract(class_ind, pred_arr)
    class_min_pred = pred_arr
    #print(label)
    y_delta = np.subtract(label, class_min_pred)
    ajd_w = np.zeros(pred_arr.size)
    copy_weights = np.zeros(pred_arr.shape)
    copy_weights = np.array(model["weights"])
    copy_weights_hold = np.array(model["weights"])
    rowindex = 0
    if(prediciton == label):
        return True
    for feat in data:
        #print(copy_weights)
        copy_weights[rowindex,:] = feat * alpha[rowindex,:]
        #print(copy_weights)
        #copy_weights[rowindex,label] = copy_weights[rowindex,label] * -1
        #print(copy_weights)
        rowindex=rowindex+1
        #ajd_w = feat *
    rowindex = 0
    copy_weights[:,label] = copy_weights[:,label] *-1
    model["weights"] = copy_weights_hold - copy_weights
    return False
    #print(label - prediciton)
def log_reg_train(data, labels, model, check_gradient=False):
    """
    Train a linear classifier by maximizing the logistic likelihood (minimizing the
negative log logistic likelihood)
    :param data: size (n, m) ndarray containing m examples described by n features
each
    :type data: ndarray
    :param labels: length n array of the integer class labels
    :type labels: array
    :param model: dictionary containing 'weights' key. The value for the 'weights'
key is a size
        (n, num_classes) ndarray
    :type model: dict
    :param check_gradient: Boolean value indicating whether to run the numerical
gradient check, which will skip

learning after checking the gradient on the initial
model weights.
    :type check_gradient: Boolean
    :return: the learned model
    :rtype: dict
    """
    n, m = data.shape
    weights = model['weights'].ravel()
    
    def log_reg_nl(new_weights):
        """
        This internal function returns the negative log-likelihood (nl) as well as
        the gradient of the nl
        :param new_weights: weights to use for computing logistic regression
        likelihood
        :type new_weights: ndarray
        :return: tuple containing (<negative log likelihood of data>, gradient)
        :rtype: float
        """
        # reshape the weights, which the optimizer prefers to be a vector, to the more convenient matrix form
        new_weights = new_weights.reshape((n,-1))
        num_classes = np.shape(new_weights)[1]

        value = new_weights.T.dot(data)
        nll = np.zeros(new_weights.T.dot(data).shape)
        negLog = np.zeros(np.exp(new_weights.T.dot(data)).shape)
        for entry in data[0,:]:
            for classes in new_weights[0,:]:
                negLog = negLog + np.exp(new_weights.T.dot(data))
            negLog = np.log(negLog)
            nll = nll + new_weights.T.dot(data) - negLog
        nl = nll * -1

        gradient = np.zeros(new_weights.T.dot(data).shape)
        negLog = np.zeros(np.exp(new_weights.T.dot(data)).shape)
        for entry in data[0,:]:
            for classes in new_weights[0,:]:
                negLog = negLog + np.exp(new_weights.T.dot(data))
            gradient = np.divide(np.exp(new_weights.T.dot(data)), negLog)
# compute the gradient
        return nl, gradient
    if check_gradient:
        grad_error = check_grad(lambda w: log_reg_nl(w)[0], lambda w: log_reg_nl(w)
[1].ravel(), weights)
        print("Provided gradient differed from numerical approximation by %e (should be around 1e-3 or less)" % grad_error)
        return model
# pass the internal objective function into the optimizer

    res = minimize(lambda w: log_reg_nl(w)[0], jac=lambda w: log_reg_nl(w)
[1].ravel(), x0=weights, method='BFGS')
    weights = res.x
    model = {'weights': weights.reshape((n, -1))}
    return model

def plot_predictions(data, labels, predictions):
    """
    Utility function to visualize 2d, 4-class data
    :param data:
    :type data:
    :param labels:
    :type labels:
    :param predictions:
    :type predictions:
    :return: list of artists that can be used for plot management
    :rtype: list
    """
    num_classes = np.unique(labels).size
    markers = ['x', 'o', '*', 'd']
    artists = []
    for i in range(num_classes):
        artists += plt.plot(data[0, np.logical_and(labels == i, labels == predictions)], data[1, np.logical_and(labels == i, labels == predictions)], markers[i] + 'g')
        artists += plt.plot(data[0, np.logical_and(labels == i, labels != predictions)], data[1, np.logical_and(labels == i, labels != predictions)], markers[i] + 'r')
    return artists

def logsumexp(matrix, dim=None):
    """
    Compute log(sum(exp(matrix), dim)) in a numerically stable way.
    :param matrix: input ndarray
    :type matrix: ndarray
    :param dim: integer indicating which dimension to sum along
    :type dim: int
    :return: numerically stable equivalent of np.log(np.sum(np.exp(matrix), dim)))
    :rtype: ndarray
    """
    try:
        with np.errstate(over='raise', under='raise'):
            return np.log(np.sum(np.exp(matrix), dim, keepdims=True))
    except:
        max_val = np.nan_to_num(matrix.max(axis=dim, keepdims=True))
        with np.errstate(under='ignore', divide='ignore'):
            return np.log(np.sum(np.exp(matrix - max_val), dim, keepdims=True)) + max_val