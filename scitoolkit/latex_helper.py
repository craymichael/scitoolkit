from scitoolkit.py23 import *

from scitoolkit.model_evaluation.metrics_helper import eval_metric


def gen_latex_confusion_matrix_str(cm=None, y_true=None, y_pred=None,
                                   lookup=None, normalize=False):
    if ((cm is None and y_true is None) or
            (y_true is not None and y_pred is None) or
            (cm is not None and y_pred is not None)):
        raise ValueError('Either `cm` or both `y_true` and `y_pred` '
                         'must be specified.')

    if cm is None:
        cm = eval_metric('confusion_matrix', y_true, y_pred)

    if lookup is None:
        lookup = map(str, range(len(cm)))

    if normalize:
        cm = cm / cm.sum()

    return ('\\\\\n\\hline\n'.join(['', '&'.join(lookup)] +
                                   ['&'.join([lookup[i]] +
                                             list(map(lambda s: '{:0.3f}'.format(s),
                                                      r)))
                                    for i, r in enumerate(cm)] + ['']
                                   ))
