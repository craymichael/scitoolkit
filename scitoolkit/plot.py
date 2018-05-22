# =====================================================================
# plot.py - A scitoolkit file
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

from scitoolkit.system.sys_helper import sys_has_display

if not os.name == 'nt' or not sys_has_display():  # TODO best place for this
    import matplotlib as mpl

    mpl.use('Agg')
    del mpl

import matplotlib.pyplot as plt
import numpy as np
from functools import wraps

from scitoolkit.model_evaluation.metrics_helper import eval_metrics, eval_metric

plt.rcParams.update({  # TODO this+sns config....and best place for this...
    'figure.figsize': '6.4, 4.8',  # inches
    'figure.dpi': 150,
    'savefig.format': 'png',
    'font.size': 10,
    'font.family': 'Times New Roman'
})

PLT_WRAPPER_ATTRS = ['title', 'xlabel', 'ylabel', 'xlim', 'ylim']
PLT_WRAPPER_ATTRS_3D = ['zlabel', 'zlim']


# TODO make generic for n axes with subplot shape
# TODO for grid, use fig.suptitle for 1 title, otherwise per-plot titles (check dims)
def plot_wrapper(_3d=False, **plt_kwargs):
    """
    see PLT_WRAPPER_ATTRS
    """

    def plot_decorator(func):
        @wraps(func)
        def call(*args, savefig=False, format=None, **kwargs):
            plt.figure()  # TODO (see above note)
            # Call plot function
            r = func(*args, **kwargs)
            if type(r) is tuple:
                ret = r[1]
                r = r[0]
            else:
                ret = None
            # title
            title = kwargs.get('title') or plt_kwargs.get('title')
            if title is not None:
                plt.title(title)

            def _process_plt_attrs(_attrs):
                for attr in _attrs:
                    plt_func = getattr(plt, attr)
                    value = kwargs.get(attr) or plt_kwargs.get(attr)
                    if value is not None:
                        if r is not None and attr in r:
                            # Perform string formatting of specified results
                            # TODO check if str, allow other options here...
                            for str_f in r[attr]:
                                value = value.format(str_f)
                        plt_func(value)

            _process_plt_attrs(PLT_WRAPPER_ATTRS)

            if _3d:
                _process_plt_attrs(PLT_WRAPPER_ATTRS_3D)

            # save plot
            if savefig is not False:
                if savefig is True:
                    savefig = func.default_file_name
                if format is None:
                    format = plt.rcParams['savefig.format']
                if savefig[-len(format):].lower() != format:
                    savefig += '.' + format
                plt.savefig(savefig, format=format)
            # display plot
            plt.show()

            return ret

        return call

    return plot_decorator


# TODO:
# - Grid plots (seaborn?)


def gen_good_random_cmap(n, colorspace='hsv'):
    cmap = plt.cm.get_cmap(colorspace, n)
    for idx in np.random.permutation(np.arange(n)):
        yield cmap(idx)


def _plot_curve(x_vals, y_vals, color='b'):
    plt.step(x_vals, y_vals, color=color, alpha=0.2,
             where='post')
    plt.fill_between(x_vals, y_vals, step='post', alpha=0.2,
                     color=color)


def _check_classification_args(y_true, y_pred):
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    shape = y_pred.shape
    if shape != y_true.shape:
        raise ValueError('Shapes of `y_true` ({}) and `y_pred` ({}) '
                         'are not the same.'.format(y_true.shape, y_pred.shape))
    if len(shape) > 2:
        raise ValueError('Shape of labels cannot be > 2: {}'.format(shape))
    elif len(shape) == 2:
        n_classes = shape[1]
    else:
        n_classes = None
    return y_true, y_pred, n_classes


@plot_wrapper(  # TODO backend matplotlib or sns
    xlabel='Recall',
    xlim=[0.0, 1.0],
    ylabel='Precision',
    ylim=[0.0, 1.05],
    title='{:d}-class Precision-Recall curve: AP={:0.3f}'
)
def plot_pr_curve(y_true, y_pred, **kwargs):
    y_true, y_pred, n_classes = _check_classification_args(y_true, y_pred)
    # Eval PR metrics
    average_precision = eval_metric('average_precision', y_true, y_pred)
    if n_classes is None:
        n_classes = 2  # Binary
        precision, recall, thresholds = eval_metrics('pr_curve',
                                                     y_true, y_pred)
        # Setup plot
        _plot_curve(recall, precision)
    else:
        cmap = gen_good_random_cmap(n_classes)
        for c in cmap:
            precision, recall, thresholds = eval_metrics('pr_curve',
                                                         y_true, y_pred)
            # Setup plot
            _plot_curve(recall, precision, color=c)

    return dict(title=[n_classes, average_precision]), average_precision


plot_pr_curve.default_file_name = 'precision-recall_curve'


@plot_wrapper(
    xlabel='TPR',
    xlim=[0.0, 1.0],
    ylabel='FPR',
    ylim=[0.0, 1.05],
    title='{:d}-class ROC curve'
)
def plot_roc_curve(y_true, y_pred, **kwargs):
    y_true, y_pred, n_classes = _check_classification_args(y_true, y_pred)
    if n_classes is None:
        n_classes = 2  # Binary
        # Eval PR metrics
        fpr, tpr, thresholds = eval_metric('roc_curve',
                                           y_true, y_pred)
        # Setup plot
        _plot_curve(tpr, fpr)
    else:
        cmap = gen_good_random_cmap(n_classes)
        for c in cmap:
            # Eval PR metrics
            fpr, tpr, thresholds = eval_metric('roc_curve',
                                               y_true, y_pred)
            # Setup plot
            _plot_curve(tpr, fpr, color=c)
    return dict(title=[n_classes])


plot_roc_curve.default_file_name = 'roc_curve'


@plot_wrapper()
def plot_image(im, cmap=None, interpolation='nearest',
               vmin=None, vmax=None, ax=None, **kwargs):
    if ax is None:  # TODO
        return None, plt.imshow(im, cmap=cmap, interpolation=interpolation,
                                vmin=vmin, vmax=vmax)
    else:
        return None, ax.imshow(im, cmap=cmap, interpolation=interpolation,
                               vmin=vmin, vmax=vmax)


def plot_heatmap(matrix_2d, normalize=False, **kwargs):
    if normalize:  # TODO other types of normalization....pass to plt?
        matrix_2d /= matrix_2d.max()
        vmin = min(0., matrix_2d.min())
        vmax = 1.
    else:
        vmin = matrix_2d.min()
        vmax = matrix_2d.max()

    fig, ax = plt.gcf(), plt.gca()  # TODO...is this really best way...(plt.subplots?)

    cax = plot_image(matrix_2d, cmap='coolwarm', interpolation='nearest',
                     vmin=vmin, vmax=vmax, ax=ax, **kwargs)
    ticks = [vmin, (vmax + vmin) / 2, vmax]
    cbar = fig.colorbar(cax, ticks=ticks)
    cbar.ax.set_yticklabels(map(str, ticks))


def plot_confusion_matrix(cm=None, y_true=None, y_pred=None, normalize=False,
                          title='Confusion Matrix', xlabel='True',
                          ylabel='Predicted', **kwargs):
    if ((cm is None and y_true is None) or
            (y_true is not None and y_pred is None) or
            (cm is not None and y_pred is not None)):
        raise ValueError('Either `cm` or both `y_true` and `y_pred` '
                         'must be specified.')

    if cm is None:
        cm = eval_metric('confusion_matrix', y_true, y_pred)

    plot_heatmap(cm, normalize=normalize, title=title, xlabel=xlabel,
                 ylabel=ylabel, **kwargs)


@plot_wrapper()
def plot_histogram(x, normalize=False, cumulative=False, **kwargs):
    plt.hist(x, density=normalize, cumulative=cumulative)


@plot_wrapper(
    _3d=True
)
def plot_3d(x, y, z, **kwargs):
    ax = plt.gcf().add_subplot(111, projection='3d')  # TODO
    ax.plot(x, y, z)


# TODO: plot factory. Allow for grid of plots of various types ('heatmap', 'cm', 'roc', 'hist', etc.....)
