# =====================================================================
# latex_helper.py - A scitoolkit file
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
