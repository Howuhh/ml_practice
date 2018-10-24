import numpy as np

from utils import split_by_column

def information_gain(column, y, threshold, criterion="gini"):
    """return infomation gain for such split"""

    y_left, y_right = split_by_column(column, y, threshold)

    left_fraction = y_left.shape[0] / column.shape[0]
    right_fraction = y_right.shape[0] / column.shape[0]

    assert isinstance(criterion, str), "criterion must be string: ('gini', 'entropy', 'variance', 'mad_median')"
    
    if criterion == "entropy":
        gain_func = entropy
    elif criterion == "variance":
        gain_func = variance
    elif criterion == "mad_median":
       gain_func = mad_median
    else:
        gain_func = gini  
        
    return gain_func(y) - left_fraction*gain_func(y_left) - right_fraction*gain_func(y_right)

def entropy(y):
    """return an entropy of a discrete vector"""
    class_fractions = np.unique(y, return_counts=True)[1] / y.shape[0]
    entropy = -np.sum(class_fractions * np.log2(class_fractions))

    return entropy

def gini(y):
    """return gini of a discrete vector"""
    class_fractions = np.unique(y, return_counts=True)[1] / y.shape[0]
    gini = 1 - np.sum(class_fractions**2) 

    return gini

def variance(y):
    """mean quadratic deviation from average"""
    return np.sum((y - np.mean(y)**2)) / y.shape[0]

def mad_median(y):
    """mean deviation from the median"""
    return np.sum(abs(y - np.median(y))) / y.shape[0]


if __name__ == "__main__":
    column = np.array([40,34,12,23,4,12,30,40,2,1,0])
    y = np.array([1,2,4,5,9,12,30,40,2,1,0])
    all_thresholds = np.linspace(0, 40, num=10)
    print([information_gain(column, y, t, criterion="gini") for t in all_thresholds])
    print("No test today :/")

    