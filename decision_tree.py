"""This module includes methods for training and predicting using decision trees."""
import numpy as np


def calculate_information_gain(data, labels):
    """
    Computes the information gain for each feature in data

    :param data: n x m matrix of n features and m examples
    :type data: ndarray
    :param labels: m x 1 vector of class labels for m examples
    :type labels: array
    :return: n x 1 vector of information gain for each feature
    :rtype: array
    """
    all_labels = np.unique(labels)
    num_classes = len(all_labels)

    class_count = np.zeros(num_classes)

    n, m = data.shape

    parent_entropy = 0
    for c in range(num_classes):
        class_count[c] = np.sum(labels == all_labels[c])
        if class_count[c] > 0:
            class_prob = class_count[c] / m
            parent_entropy -= class_prob * np.log(class_prob)

    # print("Parent entropy is %d\n" % parent_entropy)

    gain = parent_entropy * np.ones(n) #initialization of gains for every attribute


    num_x = data.dot(np.ones(m)) 
    prob_x = num_x / m # fraction of examples containing each feature
    prob_not_x = 1 - prob_x

    for c in range(num_classes):
        # print("Computing contribution of class %d." % c)
        num_y = np.sum(labels == all_labels[c])
        # this next line sums across the rows of data, multiplied by the
        # indicator of whether each column's label is c. It counts the number
        # of times each feature is on among examples with label c.
        # We again use the dot product for sparse-matrix compatibility
        data_with_label = data[:, labels == all_labels[c]]
        num_y_and_x = data_with_label.dot(np.ones(data_with_label.shape[1]))

        # Prevents Python from outputting a divide-by-zero warning
        with np.errstate(invalid='ignore'):
            prob_y_given_x = num_y_and_x / (num_x + 1e-8) # probability of observing class c for each feature
        prob_y_given_x[num_x == 0] = 0

        nonzero_entries = prob_y_given_x > 0
        if np.any(nonzero_entries):
            with np.errstate(invalid='ignore', divide='ignore'):
                children_entropy = - np.multiply(np.multiply(prob_x, prob_y_given_x), np.log(prob_y_given_x))
            gain[nonzero_entries] -= children_entropy[nonzero_entries]

        # The next lines compute the probability of y being c given x = 0 by
        # subtracting the quantities we've already counted
        # num_y - num_y_and_x is the number of examples with label y that
        # don't have each feature, and m - num_x is the number of examples
        # that don't have each feature
        with np.errstate(invalid='ignore'):
            prob_y_given_not_x = (num_y - num_y_and_x) / ((m - num_x) + 1e-8)
        prob_y_given_not_x[m - num_x == 0] = 0

        nonzero_entries = prob_y_given_not_x > 0
        if np.any(nonzero_entries):
            with np.errstate(invalid='ignore', divide='ignore'):
                children_entropy = - np.multiply(np.multiply(prob_not_x, prob_y_given_not_x), np.log(prob_y_given_not_x))
            gain[nonzero_entries] -= children_entropy[nonzero_entries]

    return gain


def decision_tree_train(train_data, train_labels, params):
    """Train a decision tree to classify data using the entropy decision criterion.

    :param train_data: n x m numpy matrix (ndarray) of n binary features for m examples
    :type train_data: ndarray
    :param train_labels: length m numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. Must include a 'max_depth' value
    :type params: dict
    :return: dictionary encoding the learned decision tree
    :rtype: dict
    """
    max_depth = params['max_depth']

    labels = np.unique(train_labels)
    num_classes = labels.size

    model = recursive_tree_train(train_data, train_labels, depth=0, max_depth=max_depth, num_classes=num_classes)
    return model


def recursive_tree_train(data, labels, depth, max_depth, num_classes):
    """Helper function to recursively build a decision tree by splitting the data by a feature.

    :param data: n x m numpy matrix (ndarray) of n binary features for m examples
    :type data: ndarray
    :param labels: length m numpy array with integer labels
    :type labels: array_like
    :param depth: current depth of the decision tree node being constructed
    :type depth: int
    :param max_depth: maximum depth to expand the decision tree to
    :type max_depth: int
    :param num_classes: number of classes in the classification problem
    :type num_classes: int
    :return: dictionary encoding the learned decision tree node
    :rtype: dict
    """


    class_list = np.unique(labels)
    num_class_mat = np.unique(labels,return_counts=True)[1]
    node = {} 

    if  num_classes == 1 or depth == max_depth:
        feat_max_index = np.argmax(num_class_mat)
        pred_from_class = class_list[feat_max_index]
        node['node_pred_split'], node['node_feat_split'] = pred_from_class, -1
    else:

        max_gain = np.argmax(calculate_information_gain(data, labels))
        node['node_feat_split'] = max_gain
        class_max = np.argmax(num_classes)
        pull_class_slice = class_list[class_max]
        calc_next_node = {'node_pred_split': pull_class_slice, 'node_feat_split': -1}

        curr_row_max_gain = data[max_gain, :]
        gain_true_mask = data[:,np.where(curr_row_max_gain == True)[0]]
        split_d_left = np.delete(gain_true_mask, max_gain, 0)

        gain_false_mask = data[:,np.where(curr_row_max_gain == False)[0]]
        split_d_right = np.delete(gain_false_mask, max_gain, 0)

        split_l_left = labels[np.where(curr_row_max_gain == True)]
        split_l_right = labels[np.where(curr_row_max_gain == False)]


        num_classes_left, num_classes_right = np.unique(split_l_left).size, np.unique(split_l_right).size

        updated_depth = depth+1

        node['left'] = calc_next_node if split_l_left.size == 0 else recursive_tree_train(split_d_left, split_l_left, updated_depth, max_depth, num_classes_left)
        node['right'] = calc_next_node if split_l_right.size == 0 else recursive_tree_train(split_d_right, split_l_right, updated_depth, max_depth, num_classes_right)


    return node




def decision_tree_predict(data, model):
    """Predict most likely label given computed decision tree in model.

    :param data: n x m ndarray of n binary features for m examples.
    :type data: ndarray
    :param model: learned decision tree model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    data_tag, num_data=data.shape
    mat_l=np.zeros(num_data)
    all_data = data[:,:]
    total_tree = range(num_data)
    curr_tree=model
    for tree_index in total_tree:
        curr_tree=model
        data_row=all_data[:,tree_index]
        leaf_node = False
        while(not leaf_node):
            if curr_tree['node_feat_split']!=-1:
                if data_row[curr_tree['node_feat_split']]==True:
                    data_row=np.delete(data_row,curr_tree['node_feat_split']) 
                    curr_tree=curr_tree['left']
                else:
                    data_row=np.delete(data_row,curr_tree['node_feat_split'])
                    curr_tree=curr_tree['right']
            else:
                leaf_node = True
                mat_l[tree_index]=curr_tree['node_pred_split']

    return mat_l
