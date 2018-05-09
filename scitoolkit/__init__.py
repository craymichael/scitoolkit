from scitoolkit.py23 import *

import lazy_import  # TODO benchmark with and without this module in use...

EXPENSIVE_MODULES = [
    'numpy',
    'tensorflow',
    'keras',
    'sklearn',
    'scipy',
    'skimage',
    'matplotlib',
    'seaborn'
]

for m in EXPENSIVE_MODULES:
    lazy_import.lazy_module(m)

from scitoolkit.np_helper import (get_dtype, get_type, min_int_dtype, min_float_dtype,
                                  min_uint_dtype, min_complex_dtype, min_dtype, is_complex,
                                  is_float, is_int, is_uint)
from scitoolkit.py_helper import (is_str, is_py3, is_py2, filter_unused_kwargs,
                                  can_reverse_dict, reverse_dict, hashable)
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
