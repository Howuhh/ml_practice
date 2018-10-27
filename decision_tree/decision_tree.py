# assert statement is all you need :p
import random
import numbers

import numpy as np
import pandas as pd

from utils import split_dataset
from sklearn.base import BaseEstimator
from gain_functions import information_gain

RANDOM_STATE = 17

class Node():
    
    def __init__(self, feature_idx=0, threshold=0, labels=None, left=None, right=None, gain=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        
        self.left = left
        self.right = right
        
        self.gain = gain
        self.node_prediction = None 
        self.node_prob_prediction = None
    
    @property
    def is_last(self):
        return not bool(self.left and self.right)

class DecisionTree(BaseEstimator):
    """simple recursive implementation of decision tree for 3 ODS's hw"""
    def __init__(self, criterion="gini", debug=False, max_depth=None, min_samples_split=10, max_features=None, random_state=None):
        self.node = None 
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.debug = debug
        self.random_state = random_state
        self.x_columns = None
        self.n_clasess = None
       
    def _find_all_splits(self, column):
        """find all possible threshold values by given column"""
        all_thresholds = np.empty(len(column) - 1)

        all_values = np.unique(column)
        for index in range(len(all_values) - 1):
            threshold = (all_values[index] + all_values[index + 1]) / 2
            all_thresholds[index] = threshold

        return all_thresholds

    def _find_best_split(self, X, y):
        """find best split -> return column index, max gain and optimal threshold"""

        assert isinstance(X, np.ndarray), "X must be ndarray type"
        assert isinstance(y, np.ndarray), "y must be ndarray type"
        
        if not (self.max_features is None):
            assert isinstance(self.max_features, numbers.Complex), "max features must be a number" 

            assert self.max_features > 0, "max features must be > 0.0"
            assert self.max_features <= X.shape[1], "max fearues must be <= X columns"
        
        # sample features for split + set seed
        if (self.max_features is None):
            self.max_features = X.shape[1]
        else:
            self.max_features = round(self.max_features)
                 
        if self.random_state is None:
            self.random_state = random.uniform(1, 2*32) 

        assert isinstance(self.random_state, numbers.Complex)
        
        random.seed(self.random_state)
        # TODO: is this right? in an ordinary tree, features are selected once!
        # if base class for random forest -> select features for every split
        if self.x_columns is None:
            self.x_columns = random.sample(range(0, X.shape[1]), self.max_features)
       
        # node attributes
        max_gain, column_idx, threshold = None, None, None

        for col_idx in self.x_columns:
            # get column and all thresholds for it
            column = X[:, col_idx]
            all_thresholds = self._find_all_splits(column)
            # information gain for all thresholds
            inf_gain_by_split = np.array([information_gain(column, y, t, criterion=self.criterion) for t in all_thresholds])
            gain = inf_gain_by_split[inf_gain_by_split.argmax()]
            best_threshold = all_thresholds[inf_gain_by_split.argmax()]

            if (max_gain is None) or (gain > max_gain):
                max_gain, column_idx, threshold = gain, col_idx, best_threshold
        
        return max_gain, column_idx, threshold 

    def _predict_prob_from_leaf(self, y):
        # classes should be sorted and labeled from 0 to N! yeah, not very realistic, but...
        class_prob = np.zeros(self.n_clasess)
        classes_from_leaf = np.unique(y, return_counts=True)

        # for all unique classes in leaf
        for cls_index, class_ in enumerate(classes_from_leaf[0]):
            # get class fraction
            class_prob[class_] = classes_from_leaf[1][cls_index] / float(y.shape[0])      
        return class_prob

    def _predict_from_leaf(self, y):
        """make final prediction for leaf"""
        if (self.criterion == "variance") or (self.criterion == "mad_median"):
            self.node.node_prediction = np.mean(y)
        else:
            assert y.shape[0] > 0, "y is 0 len"
            self.node.node_prediction = np.bincount(y).argmax()

            # TODO: zero if not in this leaf
            self.node.node_prob_prediction = self._predict_prob_from_leaf(y)


    def fit(self, X, y):
        """fit tree in X, y"""

        # TODO: add n_classes attribute
        try:
            # only for numpy arrays for now
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if self.max_depth is None:
                # The absolute maximum depth would be N−1, where N is the number of training samples. 
                # https://stats.stackexchange.com/questions/65893/maximal-depth-of-a-decision-tree
                self.max_depth = X.shape[0] - 1 
            if self.n_clasess is None:
                self.n_clasess = len(set(y))

            assert (X.shape[0] > self.min_samples_split)
            if not (self.max_depth is None):
                assert (self.max_depth > 0)
            
            gain, column_idx, threshold = self._find_best_split(X, y)
            X_left, X_right, y_left, y_right = split_dataset(X, y, column=column_idx, t=threshold)

            self.node = Node(feature_idx=column_idx, threshold=threshold, labels=y, gain=gain)

            # build left and right child for max
            self.node.left = DecisionTree(criterion=self.criterion, debug=self.debug, max_depth=self.max_depth - 1)
            # if base class for random forest -> remove that attribute
            self.node.left.x_columns = self.x_columns
            self.node.left.n_clasess = self.n_clasess
            self.node.left.fit(X_left, y_left)
            

            self.node.right = DecisionTree(criterion=self.criterion, debug=self.debug, max_depth=self.max_depth - 1)
            # if base class for random forest -> remove that attribute
            self.node.right.x_columns = self.x_columns
            self.node.right.n_clasess = self.n_clasess
            self.node.right.fit(X_right, y_right)
        # not the best idea, it is impossible to check for other conditions with assert :(
        except AssertionError:
            self.node = Node()
            self._predict_from_leaf(y)
            # test info about predictions
            if self.debug:
                print("Is Last Node: ", self.node.is_last)
                print("Data shapes: ", X.shape, y.shape)
                print("Y: ", y)
                print("Prediction: ", self.node.node_prediction)
                print("Predict proba: ", self.node.node_prob_prediction)
            return self

    def _predict_by_row(self, row, prob=False):
        assert row.shape[0] > 0, "empty row"
        # is it a slow way??

        if not self.node.is_last: 
            if row[self.node.feature_idx] < self.node.threshold:
                return self.node.left._predict_by_row(row, prob)
            else:
                return self.node.right._predict_by_row(row, prob)
        
        if prob:
            return self.node.node_prob_prediction  
        return self.node.node_prediction

    def predict(self, X):
        """make predictons for given data"""
        assert isinstance(X, np.ndarray), "X must be numpy array"      
        n_rows, _ = X.shape
        predictions = np.empty(n_rows)

        for row in range(n_rows):
            row_pred = self._predict_by_row(X[row, :])
            predictions[row] = row_pred
        return predictions

    def predict_proba(self, X):
        # TODO: incorrect return format, should be array [n_clasess] x 1, zero if class not in leaf
        assert isinstance(X, np.ndarray), "X must be numpy array"      
        n_rows, _ = X.shape
        # predictions = np.empty(n_rows, 2)
        predictions = []
        for row in range(n_rows):
            row_pred = self._predict_by_row(X[row, :], prob=True)
            print(row_pred)
            predictions.append(row_pred)
            # predictions[row] = row_pred
        return np.array(predictions)