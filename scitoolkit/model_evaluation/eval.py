# =====================================================================
# eval.py - A scitoolkit file
# Copyright (C) 2018  Zach Carmichael
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# =====================================================================
from scitoolkit.util.py23 import *

from sklearn.base import BaseEstimator as SKBaseEstimator
from sklearn.externals.joblib import Memory

from scitoolkit.util.py_helper import is_str, hasmethod
from scitoolkit.model_evaluation.metrics_helper import eval_metrics
from scitoolkit.model_evaluation.cv import get_cv
from scitoolkit.models.base import BaseModel

DEFAULT_MODEL_TRAIN_FUNC = 'train'
DEFAULT_MODEL_TEST_FUNC = 'test'
DEFAULT_SKLEARN_TRAIN_FUNC = 'fit'
DEFAULT_SKLEARN_TEST_PROBA_FUNC = 'predict_proba'
DEFAULT_SKLEARN_TEST_FUNC = 'predict'


# TODO cache this function...
# TODO rename to something more fitting for function that can do EITHER training or evaluation only, or BOTH...
def train_and_eval(X, y, model, train=True, test=True, train_func=None,
                   test_func=None, cv=None, iid=True,
                   return_train_score=False, metrics=None, target_metric=None,
                   time_series=False, eval_kwargs=None):
    """

    Args:
        X:
        y:
        model:
        train:
        test:
        train_func:
        test_func:
        cv:
        iid:
        return_train_score:
        metrics:
        target_metric:
        time_series:
        eval_kwargs:

    Returns:

    """
    if return_train_score:
        raise NotImplementedError  # TODO

    if not metrics or target_metric not in metrics:
        raise ValueError('Invalid specification of metrics for evaluation.')

    if target_metric is None:
        if len(metrics) == 1:  # TODO nonetype metrics...
            target_metric = metrics[0]
        else:
            raise ValueError('"target_metric" must be provided if multiple '
                             'metrics specified.')

    eval_kwargs = eval_kwargs or {}

    if type(X) is type(y) is tuple and len(X) == len(y) == 2:  # TODO this good? for manual split specification...maybe require wrapping in object...
        splits = [(X, y)]
        cv = False
    else:
        cv = get_cv(cv)
        splits = cv.split(X, y)

    # Check specified train/eval functions
    if isinstance(model, SKBaseEstimator):  # TODO you also know that it has the {set, get}_params methods...
        if train and train_func is None:
            if hasmethod(model, DEFAULT_SKLEARN_TRAIN_FUNC):
                train_func = getattr(model, DEFAULT_SKLEARN_TRAIN_FUNC)
            else:
                raise ValueError('Could not infer the train method from the '
                                 'specified sklearn model. Please specify '
                                 '"train_func" in order to train the model.')
        if test and test_func is None:
            if hasmethod(model, DEFAULT_SKLEARN_TEST_PROBA_FUNC):
                test_func = getattr(model, DEFAULT_SKLEARN_TEST_PROBA_FUNC)
            elif hasmethod(model, DEFAULT_SKLEARN_TEST_FUNC):
                test_func = getattr(model, DEFAULT_SKLEARN_TEST_FUNC)
            else:
                raise ValueError('Could not infer the predict method from the '
                                 'specified sklearn model. Please specify '
                                 '"test_func" in order to test the model.')
    elif isinstance(model, BaseModel):
        raise NotImplementedError  # TODO
    else:
        if train and train_func is None:
            # Take a stab at it
            if hasmethod(model, DEFAULT_MODEL_TRAIN_FUNC):
                train_func = getattr(model, DEFAULT_MODEL_TRAIN_FUNC)
            else:
                raise ValueError('Could not infer model train function: please '
                                 'specify "train_func" in order to train the '
                                 'model.')
        if test and test_func is None:
            # Take a stab at it
            if hasmethod(model, DEFAULT_MODEL_TEST_FUNC):
                test_func = getattr(model, DEFAULT_MODEL_TEST_FUNC)
            else:
                raise ValueError('Could not infer model test function: please '
                                 'specify "test_func" in order to test the '
                                 'model.')

    if is_str(train_func):
        train_func = getattr(model, train_func)
    if is_str(test_func):
        test_func = getattr(model, test_func)

    test_score = 0.
    test_scores = None
    n_test = 0

    for train_idx_or_xs, test_idx_or_ys in splits:

        if len(train_idx_or_xs) <= 0 or len(test_idx_or_ys) <= 0:
            raise ValueError('Train and test must be long enough for specified '
                             'cross validation.')
        
        if cv:
            X_train = X[train_idx_or_xs]
            y_train = y[train_idx_or_xs]
    
            X_test = X[test_idx_or_ys]
            y_test = y[test_idx_or_ys]
        else:
            X_train, X_test = train_idx_or_xs
            y_train, y_test = test_idx_or_ys

        if time_series:
            test_len = X_test.shape[2]
        else:
            test_len = len(X_test)
        
        if train_func is not None:
            # Train the model
            y_train_pred = train_func(X_train, y_train)
        if test_func is not None:
            y_test_pred = test_func(X_test)
            test_scores = eval_metrics(metrics, y_test, y_test_pred,
                                       **eval_kwargs)

            print(test_scores)  # TODO

            target_score = test_scores[target_metric]

            if iid:
                test_score += target_score * test_len
                n_test += test_len
            else:
                test_score += target_score
                n_test += 1

    if n_test == 0:
        raise ValueError('No evaluation was done (n_test = 0).')

    test_score /= n_test

    return test_score, test_scores  # TODO
