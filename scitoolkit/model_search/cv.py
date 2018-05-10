# =====================================================================
# cv.py - A scitoolkit file
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
from scitoolkit.py23 import *  # py2/3 compatibility

from sklearn.model_selection import (RepeatedKFold, RepeatedStratifiedKFold,
                                     check_cv)
from sklearn.utils.multiclass import type_of_target

from scitoolkit.np_helper import is_int
from scitoolkit.py_helper import is_str


def get_cv(cv=3, n_repeats=None, y=None, classification=None,
           random_state=None):
    """Input checker utility for building a cross-validator

    Args:
        cv: int, cross-validation generator or an iterable
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
            - None, to use the default 3-fold cross-validation,
            - integer, to specify the number of folds.
            - An object to be used as a cross-validation generator.
            - An iterable yielding train/test splits.
            For integer/None inputs, if classification is True and ``y`` is
            either binary or multiclass, `StratifiedKFold` is used. In all
            other cases, `KFold` is used.
        n_repeats:
            The number of times to repeat splits.
        y:  The target variable for supervised learning problems.
        classification:
            Whether the task is a classification task, in which case stratified
            KFold will be used. Infers based on `y` if specified and
            `classification` is None.
        random_state:
            Random state to be used to generate random state for each
            repetition.

    Returns:
        A cross-validator instance that generates the train/test splits via
        its `split` method.
    """
    if cv is None:
        cv = 3
    if n_repeats is None:
        n_repeats = 1

    if is_int(cv, ignore_unsigned=False):
        # Infer classification if None and y is provided
        if classification is None:
            if ((y is not None) and
                    (type_of_target(y) in ('binary', 'multiclass'))):
                classification = True
            else:
                classification = False
        # Select appropriate CV type
        if classification:
            return RepeatedStratifiedKFold(n_splits=cv, n_repeats=n_repeats,
                                           random_state=random_state)
        else:
            return RepeatedKFold(n_splits=cv, n_repeats=n_repeats,
                                 random_state=random_state)

    # For iterables and error-handling (rely on sklearn)
    if not hasattr(cv, 'split') or is_str(cv):
        return check_cv(cv)

    return cv
