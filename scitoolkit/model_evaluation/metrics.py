from scitoolkit.py23 import *  # py2/3 compatibility

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
