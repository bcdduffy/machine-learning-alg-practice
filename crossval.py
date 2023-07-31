"""This module includes utilities to run cross-validation on general supervised learning methods."""
from __future__ import division
import numpy as np


def cross_validate(trainer, predictor, all_data, all_labels, folds, params):
    """Perform cross validation with random splits.

    :param trainer: function that trains a model from data with the template
             model = function(all_data, all_labels, params)
    :type trainer: function
    :param predictor: function that predicts a label from a single data point
                label = function(data, model)
    :type predictor: function
    :param all_data: n x m data matrix
    :type all_data: numpy ndarray
    :param all_labels: m x 1 label vector
    :type all_labels: numpy array
    :param folds: number of folds to run of validation
    :type folds: int
    :param params: auxiliary variables for training algorithm (e.g., regularization parameters)
    :type params: dict
    :return: tuple containing the average score and the learned models from each fold
    :rtype: tuple
    """
    model_fill = []
    all_folds = range(folds)
    col, row = all_data.shape
    rowlen = range(row)
    row_fold_mat = np.divide(row, folds)
    find_len = np.dot(np.ceil(row_fold_mat),folds)
    new_rowlon_mat = np.array(rowlen)
    len_array_transp = np.array(rowlen).size
    sub_arr = np.subtract(find_len, len_array_transp)
    inv_find = -np.ones(sub_arr)

    mask_col = np.append(new_rowlon_mat, inv_find).reshape((np.ceil(row_fold_mat), folds))

    all_score_mat = np.zeros(folds)


    for fold_ind in all_folds:
        train_indices = np.delete(mask_col, fold_ind, 1).ravel()
        first_val_ind = mask_col[:, fold_ind]
        takeIndexes = first_val_ind.ravel()

        all_val_labels = all_labels[takeIndexes[takeIndexes >= 0]]
        trains = train_indices[train_indices >= 0]
        for_data = all_data[:, trains]
        take_labels = all_labels[trains]

        model = trainer(for_data, take_labels, params)
        model_fill.append(model)
        totalmean = np.mean(predictions == all_val_labels)

        predictions = predictor(all_data[:, takeIndexes[takeIndexes >= 0]], trainer(for_data, take_labels, params))
        predictions = predictions[0] if type(predictions) == tuple else predictions

        all_score_mat[fold_ind] = totalmean

    return np.mean(all_score_mat), model_fill
