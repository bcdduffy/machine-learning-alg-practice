"""This module includes methods for training and predicting using naive Bayes."""
import numpy as np


def naive_bayes_train(train_data, train_labels, params):
    """Train naive Bayes parameters from data.

    :param train_data: n x m numpy matrix (ndarray) of n binary features for m examples
    :type train_data: ndarray
    :param train_labels: length m numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. (Optional. Can be empty)
    :type params: dict
    :return: model learned with the priors and conditional probabilities of each feature
    :rtype: model
    """
    labels = np.unique(train_labels)
    n, m = train_data.shape
    num_classes = labels.size
    result = np.unique(train_labels,return_counts=True)[1]

    model={}

    prior_dist = np.divide(result,m)
    model['feat_pr_prob']=prior_dist

    tru_features=np.zeros((num_classes,n))
    false_features=np.zeros((num_classes,n))
    classes_arr = range(num_classes)

    laplace_den_offset = 2
    laplace_num_offset = 1
    for class_index in classes_arr:

        data_at_feature = train_data[:,np.where(train_labels==class_index)[0]]

        number_count_feat=np.count_nonzero(data_at_feature,axis=1)
        cond_prob_num_count = number_count_feat + laplace_num_offset
        inv_cond_prob = result[class_index]-number_count_feat
        inv_cond_prob_smooth = inv_cond_prob + laplace_num_offset
        cond_prob_total_num = result[class_index] + laplace_den_offset
        feat_row_true_prob = np.divide(cond_prob_num_count, cond_prob_total_num)
        feat_row_false_prob = np.divide(inv_cond_prob_smooth, cond_prob_total_num)

        tru_features[class_index,:]=feat_row_true_prob
        false_features[class_index,:]=feat_row_false_prob


    model['tru_feat_cond_prob']=tru_features
    model['false_feat_cond_prob']=false_features


    return model

########################################################################################################################



def naive_bayes_predict(data, model):
    """Use trained naive Bayes parameters to predict the class with highest conditional likelihood.

    :param data: n x m numpy matrix (ndarray) of n binary features for m examples
    :type data: ndarray
    :param model: learned naive Bayes model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    axes_arr = 0,1,0

    list_true_prob = model['tru_feat_cond_prob']
    tru_log = np.log(list_true_prob)
    true_prob_data=tru_log.dot(data)
    true_prob_data=np.transpose(true_prob_data)

    list_false_prob = model['false_feat_cond_prob']
    false_log = np.log(list_false_prob)
    
    list_false_data=false_log.dot(np.where(data==0,1,0))
    list_false_data=np.transpose(list_false_data)

    log_prior = np.log(model['feat_pr_prob'])
    
    sum = true_prob_data+list_false_data +log_prior

    return np.argmax(sum[:,:], axis=1)
