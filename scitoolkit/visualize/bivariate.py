# =====================================================================
# bivariate.py - A scitoolkit file
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

import seaborn as sns
from scipy import stats

DEFAULT_STAT = stats.pearsonr


# TODO NOTE: These plots have {x, y}lim, kwargs, colors, etc. that can be abstracted...
def scatter(x, y, data=None, tight=False, stat_func=DEFAULT_STAT):
    # NOTE: stat_func=None results in no printed metrics
    # TODO this function can take DF and use label names for axes (also if figure-based function... so deal with that)
    if tight:
        space = 0
    else:
        space = 0.2
    return sns.jointplot(x, y, data=data, kind='scatter', space=space,
                         stat_func=stat_func)


def reg(x, y, data=None, tight=False, stat_func=DEFAULT_STAT):
    # TODO this function can take DF and use label names for axes (also if figure-based function... so deal with that)
    if tight:
        space = 0
    else:
        space = 0.2
    return sns.jointplot(x, y, data=data, kind='reg', space=space,
                         stat_func=stat_func)


def resid(x, y, data=None, tight=False, stat_func=DEFAULT_STAT):
    # TODO this function can take DF and use label names for axes (also if figure-based function... so deal with that)
    if tight:
        space = 0
    else:
        space = 0.2
    return sns.jointplot(x, y, data=data, kind='resid', space=space,
                         stat_func=stat_func)


def kde(x, y, data=None, tight=False, stat_func=DEFAULT_STAT):
    # TODO this function can take DF and use label names for axes (also if figure-based function... so deal with that)
    if tight:
        space = 0
    else:
        space = 0.2
    return sns.jointplot(x, y, data=data, kind='kde', space=space,
                         stat_func=stat_func)


def hex(x, y, data=None, tight=False, stat_func=DEFAULT_STAT):
    # TODO this function can take DF and use label names for axes (also if figure-based function... so deal with that)
    if tight:
        space = 0
    else:
        space = 0.2
    # About equivalent to:
    # f, ax = plt.subplots(figsize=(6, 6))
    # sns.kdeplot(df.x, df.y, ax=ax)
    # sns.rugplot(df.x, color="g", ax=ax)
    # sns.rugplot(df.y, vertical=True, ax=ax);
    with sns.axes_style('white'):
        return sns.jointplot(x, y, data=data, kind='hex', space=space,
                             stat_func=stat_func)
