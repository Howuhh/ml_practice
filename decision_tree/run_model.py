from decision_tree import DecisionTree

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score

def main(debug=False):
    data = load_breast_cancer()
    X, y = data["data"], data["target"]

    clf = DecisionTree(debug=debug, max_depth=10)

    print(X.shape, y.shape)

    print("Calling fit method!")
    clf.fit(X, y)
    print("Calling predict method!")
    y_pred = clf.predict_proba(X)
    print(roc_auc_score(y, y_pred[:, 1]))
    # print(y_pred)

if __name__ == "__main__":
    main(debug=False)