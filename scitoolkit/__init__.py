# =====================================================================
# __init__.py - A scitoolkit file
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

import lazy_import  # TODO benchmark with and without this module in use...

LAZY_MODULES = [
    'numpy',
    'tensorflow',
    'keras',
    'sklearn',
    'scipy',
    'skimage',
    'matplotlib',
    'seaborn'
]

for m in LAZY_MODULES:
    lazy_import.lazy_module(m)

from scitoolkit.system.sys_helper import sys_has_display

import os
if not os.name == 'nt' or not sys_has_display():  # TODO best place for this
    import matplotlib as mpl
    if mpl.get_backend() != 'agg':
        mpl.use('Agg')
    del mpl  # TODO...

from scitoolkit.util.np_helper import (get_dtype, get_type, min_int_dtype, min_float_dtype,
                                       min_uint_dtype, min_complex_dtype, min_dtype, is_complex,
                                       is_float, is_int, is_uint)
from scitoolkit.plot import (plot_pr_curve, plot_roc_curve, plot_wrapper,
                             gen_good_random_cmap)
from scitoolkit.model_evaluation.metrics_helper import eval_metrics, eval_metric

# np_helper
__all__ = ['get_dtype', 'get_type', 'min_int_dtype', 'min_float_dtype',
           'min_uint_dtype', 'min_complex_dtype', 'min_dtype', 'is_complex',
           'is_float', 'is_int', 'is_uint']
# py_helper
__all__.extend(['is_str', 'is_py3', 'is_py2', 'filter_unused_kwargs',
                'can_reverse_dict', 'reverse_dict', 'hashable'])
# plot
__all__.extend(['plot_pr_curve', 'plot_roc_curve', 'plot_wrapper', 'gen_good_random_cmap'])
# metrics
__all__.extend(['mean_per_class_accuracy'])
# metrics_helper
__all__.extend(['eval_metrics', 'eval_metric'])
