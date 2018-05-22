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

import six
import abc
from scitoolkit.util.py_helper import is_str
from scitoolkit.model_evaluation.metrics_helper import eval_metrics
from scitoolkit.model_evaluation.cv import get_cv
from sklearn.externals.joblib import Memory


# TODO cache this function...
def train_and_eval(X, y, model, train_func='train', test_func='test', cv=None,
                   iid=True, return_train_score=False, metrics=None,
                   target_metric=None):
    if return_train_score:
        raise NotImplementedError

    if target_metric is None:
        if len(metrics) == 1:  # TODO nonetype metrics...
            target_metric = metrics[0]
        else:
            raise ValueError('"target_metric" must be provided if multiple '
                             'metrics specified.')

    cv = get_cv(cv)

    if train_func is not None and is_str(train_func):
        train_func = getattr(model, train_func)
    if is_str(test_func):
        test_func = getattr(model, test_func)

    test_score = None
    n_test = 0

    for train_idx, test_idx in cv.split(X, y):
        if len(train_idx) <= 0 or len(test_idx) <= 0:
            raise ValueError('Train and test must be long enough for specified '
                             'cross validation.')
        
        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]
        
        if train_func is not None:
            # Train the model
            y_train_pred = train_func(X_train, y_train)
        if test_func is not None:
            y_test_pred = test_func(X_test, y_test)
            test_scores = eval_metrics(metrics)
            target_score = test_scores[target_metric]

            if iid:
                test_score += target_score * len(test_idx)
                n_test += len(test_idx)
            else:
                test_score += target_score
                n_test += 1

    if n_test == 0:
        raise ValueError('No evaluation was done (n_test = 0).')

    test_score /= n_test
    return test_score  # TODO other metrics?


class Model(six.with_metaclass(abc.ABCMeta, object)):
    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def test(self):
        pass
