# =====================================================================
# univariate.py - A scitoolkit file
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

import seaborn as sns


def visualize_univariate_data(x, plot='histogram{}kde'.format(PLOT_DELIMITER)):  # TODO
    """A factory function for visualizing univariate distributions."""
    plot = plot.lower()


def histogram(x, bins=None, rug=True, kde=False, normalize=False, shade_kde=False,
              fit=None, **kwargs):  # TODO
    """

    Examples:
    https://seaborn.pydata.org/tutorial/distributions.html

    ax = sns.distplot(x, rug=True, rug_kws={"color": "g"},
                      kde_kws={"color": "k", "lw": 3, "label": "KDE"},
                      hist_kws={"histtype": "step", "linewidth": 3,
                                "alpha": 1, "color": "g"})

    Args:
        x:         Observed data.
        bins:      Specification of hist bins, or None to use Freedman-Diaconis rule.
        rug:       Whether to draw a rugplot on the support axis.
        kde:       Whether to plot a gaussian kernel density estimate (KDE).
        normalize: Whether to normalize the histogram
        shade_kde: Whether to shade the KDE.
        fit:       An object with fit method, returning a tuple that can be passed to a
                   pdf method a positional arguments following an grid of values to
                   evaluate the pdf on. You might want kde=False while using this function.
        kwargs:    Other kwargs.

    Returns:
        ax
    """
    if 'kde_kws' in kwargs:
        if isinstance(kwargs['kde_kws'], dict) and 'shade' not in kwargs['kde_kws']:
            kwargs['kde_kws']['shade'] = shade_kde
    else:
        kwargs['kde_kws'] = {'shade': shade_kde}
    return sns.distplot(x, kde=kde, rug=rug, bins=bins, norm_hist=normalize,
                        fit=fit, **kwargs)  # TODO ax=ax...
