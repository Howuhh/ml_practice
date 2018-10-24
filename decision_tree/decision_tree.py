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
    
    @property
    def is_last(self):
        return not bool(self.left and self.right)

class DecisionTree(BaseEstimator):
    """simple recursive implementation of decision tree for 3 ODS's hw"""
    def __init__(self, criterion="gini", debug=False, max_depth=None, min_samples_split=10):
        self.node = None 
        self.max_depth = None
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.debug = debug
       
    def _find_all_splits(self, column):
        """find all possible threshold values by given column"""
        all_thresholds = np.empty(len(column) - 1)

        for index in range(len(column) - 1):
            threshold = (column[index] + column[index + 1]) / 2
            all_thresholds[index] = threshold

        return all_thresholds

    def _find_best_split(self, X, y):
        """find best split -> return column index, max gain and optimal threshold"""

        assert isinstance(X, np.ndarray), "X must be ndarray type"
        assert isinstance(y, np.ndarray), "y must be ndarray type"

        max_gain, column_idx, threshold = None, None, None

        for col_idx in range(X.shape[1]):
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

    def _predict_from_leaf(self, y):
        """make final prediction for leaf"""
        if (self.criterion == "variance") or (self.criterion == "mad_median"):
            self.node.node_prediction = np.mean(y)
        else:
            assert y.shape[0] > 0, "y is 0 len"
            self.node.node_prediction = np.bincount(y).argmax()

    def fit(self, X, y):
        """fit tree in X, y"""
        try:
            # only for numpy arrays for now
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            if not isinstance(y, np.ndarray):
                y = np.array(y)

            assert y.shape[0] > 0, "y is wrong"

            assert (X.shape[0] > self.min_samples_split)
            if not (self.max_depth is None):
                assert (self.max_depth > 0)
            
            gain, column_idx, threshold = self._find_best_split(X, y)
            X_left, X_right, y_left, y_right = split_dataset(X, y, column=column_idx, t=threshold)

            self.node = Node(feature_idx=column_idx, threshold=threshold, labels=y, gain=gain)

            # build left and right child for max
            self.node.left = DecisionTree(criterion=self.criterion, debug=self.debug)
            self.node.left.fit(X_left, y_left)

            self.node.right = DecisionTree(criterion=self.criterion, debug=self.debug)
            self.node.right.fit(X_right, y_right)
        except AssertionError:
            self.node = Node()
            self._predict_from_leaf(y)
            # test info about predictions
            if self.debug:
                print("Is Last Node: ", self.node.is_last)
                print("Data shapes: ", X.shape, y.shape)
                print("Y: ", y)
                print("Prediction: ", self.node.node_prediction)
            return self

    def predict(self, X):
        """make predictons for given data"""

        # TODO: implement predictions by row??
        pass

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    data = load_iris()
    X, y = data["data"], data["target"]

    clf = DecisionTree(debug=True)

    print("Calling fit method!")
    clf.fit(X, y)