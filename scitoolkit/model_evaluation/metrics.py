# =====================================================================
# metrics.py - A scitoolkit file
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
from scitoolkit.util.py23 import *  # py2/3 compatibility

import numpy as np


def _check_ndarrays_and_squeeze(*arrays):  # TODO move to more general-purpose location
    a = arrays[0].squeeze()
    shape = a.shape
    yield a

    for i, a in enumerate(arrays[1:]):
        orig_shape = a.shape
        a = a.squeeze()
        if a.shape != shape:
            raise ValueError('Shapes must agree: {} (arg 0) and {} '
                             '(arg {}).'.format(shape, orig_shape, i + 1))
        yield a


def mean_per_class_accuracy(y_true, y_pred, n_classes=None, labels=None):
    """ Computes mean per-class accuracy

    Args:
        y_true:    The true labels
        y_pred:    The predicted labels
        n_classes: The number of classes, optional. If not provided, the number of
                   unique classes or length of `labels` if provided.
        labels:    The unique labels, optional. If not provided, unique labels are used
                   if `n_classes` not provided, otherwise range(n_classes).

    Returns:
        mean per-class accuracy
    """
    y_true, y_pred = _check_ndarrays_and_squeeze(y_true, y_pred)

    if n_classes is None:
        if labels is None:
            labels = np.unique(y_true)
        n_classes = len(labels)
    elif labels is None:
        labels = np.arange(n_classes)
    elif len(labels) != n_classes:
        raise ValueError('Number of classes specified ({}) differs from '
                         'number of labels ({}).'.format(n_classes, len(labels)))
    acc = 0.
    for c in labels:
        c_mask = (y_true == c)
        c_count = c_mask.sum()
        if c_count:  # Avoid division by 0
            # Add accuracy for class c
            acc += np.logical_and(c_mask, (y_pred == c)).sum() / c_count
        else:
            # Don't count this class in accuracy (no observations)
            n_classes -= 1
    # Mean accuracy per class
    return acc / n_classes


# TODO IoU, MaP
