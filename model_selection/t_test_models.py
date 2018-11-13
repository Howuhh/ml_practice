
import pandas as pd
import numpy as np

from copy import copy

from scipy.stats import ttest_rel

# TODO: add outlier detection -> drop on scoring
# TODO: do i really need a bonferonni correction for muptiple tests?
# TODO: delayed decorator from dask -> delay loops
# TODO: replace loops by fit_and_score function -> delay(fit_and_score)(params)
def cross_val_score_(X, y, estimator, validator, scorer, n_splits=5, n_ranges=5, random_state=42, task="regression", **kwargs):
    
    assert isinstance(X, np.ndarray), "X should be np.ndarray type"
    assert isinstance(y, np.ndarray), "y should be np.ndarray type"
    assert task in ("regression", "classification"), f"unknown task {task}, should be in (regression, classification)"
    model = copy(estimator)

    scores = []
    for range_ in range(n_ranges):
        cv = validator(n_splits = n_splits, random_state = random_state + range_,  **kwargs)
       
        folds = list(cv.split(X, y))

        for train, test in folds:
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]

            model.fit(X_train, y_train)
            if task == "regression":
                y_pred = model.predict(X_test)
            else:
                y_pred = model.predict_proba(X_test)
            score = scorer(y_test, y_pred)

            scores.append(score)

    return np.array(scores)

class T_Test_Validation(object):
    def __init__(self, validator, scorer, n_splits=5, n_ranges=10, random_state=42, task="regression"):
        self.validator = validator
        self.scorer = scorer
        self.n_splits = n_splits
        self.n_ranges = n_ranges
        self.task = task
        self.random_state = random_state 

    def test_model(self, baseline_cv_score, new_model_cv_score, confidence_value=0.05):
        test_result = ttest_rel(baseline_cv_score, new_model_cv_score)

        result = {
            "t_statistic": test_result[0],
            "p_value": test_result[1],
            "significance": test_result[1] < confidence_value,
        }

        return result

    def validation_curve(self, X, y, baseline_model, new_model, param_name, param_range):
        
        baseline_model = copy(baseline_model)
        base_scores = cross_val_score_(X, y, 
                                    estimator=baseline_model,
                                    validator=self.validator,
                                    scorer=self.scorer,
                                    n_ranges=self.n_ranges,
                                    n_splits=self.n_splits,
                                    random_state=self.random_state,
                                    shuffle=True
                                    )
        # weird loop
        t_statistics = []
        for param_value in param_range:
            model = copy(new_model)
            model.set_params(**{param_name: param_value})

            model_scores = cross_val_score_(X, y, 
                                    estimator=model,
                                    validator=self.validator,
                                    scorer=self.scorer,
                                    n_ranges=self.n_ranges,
                                    n_splits=self.n_splits,
                                    random_state=self.random_state,
                                    shuffle=True
                                    )
            t_stat = self.test_model(base_scores, model_scores)["t_statistic"]
            t_statistics.append(t_stat)

        return np.array(t_statistics)