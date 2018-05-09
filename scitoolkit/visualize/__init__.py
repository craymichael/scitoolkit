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
from scitoolkit.py23 import *

from six import itervalues
from itertools import chain
import seaborn as sns

from scitoolkit.py_helper import get_module_api, func_arg_names

sns.set(color_codes=True)  # TODO do this in more appropriate place...

# NOTE: These get_module_api calls work because sns imports all sub-module functions
#       to top module level (so be aware).
# The API for seaborn.
# >>> pprint(SNS_API)
# {'axisgrid': ['FacetGrid', 'PairGrid', 'JointGrid', 'pairplot', 'jointplot'],
#  'categorical': ['boxplot',
#                  'violinplot',
#                  'stripplot',
#                  'swarmplot',
#                  'lvplot',
#                  'pointplot',
#                  'barplot',
#                  'countplot',
#                  'factorplot'],
#  'distributions': ['distplot', 'kdeplot', 'rugplot'],
#  'matrix': ['heatmap', 'clustermap'],
#  'miscplot': ['palplot'],
#  'palettes': ['color_palette',
#               'hls_palette',
#               'husl_palette',
#               'mpl_palette',
#               'dark_palette',
#               'light_palette',
#               'diverging_palette',
#               'blend_palette',
#               'xkcd_palette',
#               'crayon_palette',
#               'cubehelix_palette',
#               'set_color_codes'],
#  'rcmod': ['set',
#            'reset_defaults',
#            'reset_orig',
#            'axes_style',
#            'set_style',
#            'plotting_context',
#            'set_context',
#            'set_palette'],
#  'regression': ['lmplot', 'regplot', 'residplot'],
#  'timeseries': ['tsplot'],
#  'utils': ['desaturate',
#            'saturate',
#            'set_hls_values',
#            'despine',
#            'get_dataset_names',
#            'load_dataset'],
#  'widgets': ['choose_colorbrewer_palette',
#              'choose_cubehelix_palette',
#              'choose_dark_palette',
#              'choose_light_palette',
#              'choose_diverging_palette']}
# (Last Updated: sns.__version__ == 0.8.1)
SNS_API = get_module_api(sns, depth=2, req_all_attr=True, skip_levels=[0],
                         exclude_childless_modules=True,
                         merge_dicts='soft')  # TODO this should be in plot file?
# Seaborn plots
# >>> pprint(SNS_API_PLOTS)
# {'axisgrid': ['pairplot', 'jointplot'],
#  'categorical': ['boxplot',
#                  'violinplot',
#                  'stripplot',
#                  'swarmplot',
#                  'lvplot',
#                  'pointplot',
#                  'barplot',
#                  'countplot',
#                  'factorplot'],
#  'distributions': ['distplot', 'kdeplot', 'rugplot'],
#  'matrix': ['heatmap', 'clustermap'],
#  'miscplot': ['palplot'],
#  'regression': ['lmplot', 'regplot', 'residplot'],
#  'timeseries': ['tsplot']}
# (Last Updated: sns.__version__ == 0.8.1)
SNS_API_PLOTS = get_module_api(sns, depth=2, req_all_attr=True, skip_levels=[0],
                               exclude_childless_modules=True,
                               filter_func=lambda name, attr: name.endswith('plot') or
                                                              name.endswith('map'),
                               merge_dicts='soft')
# Seaborn axis-based plots
# >>> pprint(SNS_API_AX)
# {'categorical': ['boxplot',
#                  'violinplot',
#                  'stripplot',
#                  'swarmplot',
#                  'lvplot',
#                  'pointplot',
#                  'barplot',
#                  'countplot'],
#  'distributions': ['distplot', 'kdeplot', 'rugplot'],
#  'matrix': ['heatmap'],
#  'regression': ['regplot', 'residplot'],
#  'timeseries': ['tsplot']}
# (Last Updated: sns.__version__ == 0.8.1)
SNS_API_AX = get_module_api(sns, depth=2, req_all_attr=True, skip_levels=[0],
                            exclude_childless_modules=True,
                            filter_func=lambda name, attr: (name.endswith('plot') or
                                                            name.endswith('map')) and
                                                           'ax' in func_arg_names(attr),
                            merge_dicts='soft')
# Seaborn fig-based plots
# See: https://stackoverflow.com/questions/35042255/how-to-plot-multiple-seaborn-jointplot-in-subplot
# >>> pprint(SNS_API_FIG)
# {'axisgrid': ['pairplot', 'jointplot'],
#  'categorical': ['factorplot'],
#  'matrix': ['clustermap'],
#  'miscplot': ['palplot'],
#  'regression': ['lmplot']}
# (Last Updated: sns.__version__ == 0.8.1)
SNS_API_FIG = get_module_api(sns, depth=2, req_all_attr=True, skip_levels=[0],
                             exclude_childless_modules=True,
                             filter_func=lambda name, attr: (name.endswith('plot') or
                                                             name.endswith('map')) and
                                                            'ax' not in func_arg_names(attr),
                             merge_dicts='soft')
# The valid sns plot names
# >>> VALID_PLOTS
# ['pairplot',
#  'jointplot',
#  'boxplot',
#  'violinplot',
#  'stripplot',
#  'swarmplot',
#  'lvplot',
#  'pointplot',
#  'barplot',
#  'countplot',
#  'factorplot',
#  'distplot',
#  'kdeplot',
#  'rugplot',
#  'heatmap',
#  'clustermap',
#  'palplot',
#  'lmplot',
#  'regplot',
#  'residplot',
#  'tsplot']
# (Last Updated: sns.__version__ == 0.8.1)
VALID_PLOTS = [plist for plist in itervalues(SNS_API_PLOTS)]
VALID_PLOTS = list(chain(*VALID_PLOTS))  # merge 2D to 1D list
# Valid plot aliases (Keys are aliases)
VALID_ALIASES = {
    'histogram': 'hist',  # TODO (hist is arg not function)
}
# Updates valid aliases the with following dict:
# {'pair': 'pairplot',
#  'joint': 'jointplot',
#  'box': 'boxplot',
#  'violin': 'violinplot',
#  'strip': 'stripplot',
#  'swarm': 'swarmplot',
#  'lv': 'lvplot',
#  'point': 'pointplot',
#  'bar': 'barplot',
#  'count': 'countplot',
#  'factor': 'factorplot',
#  'dist': 'distplot',
#  'kde': 'kdeplot',
#  'rug': 'rugplot',
#  'heat': 'heatmap',
#  'cluster': 'clustermap',
#  'pal': 'palplot',
#  'lm': 'lmplot',
#  'reg': 'regplot',
#  'resid': 'residplot',
#  'ts': 'tsplot'}
# (Last Updated: sns.__version__ == 0.8.1)
for p in VALID_PLOTS:
    if p.endswith('plot'):
        k = p[:-4]
    elif p.endswith('map'):
        k = p[:-3]
    else:
        raise ValueError('Unexpected value in `VALUE_PLOTS`: {}'.format(p))
    VALID_ALIASES[k] = p
# Delimiter TODO
PLOT_DELIMITER = '+'
# TODO
THING = {
    'distplot': ['hist', 'kde', 'rug']
}

__all__ = []
