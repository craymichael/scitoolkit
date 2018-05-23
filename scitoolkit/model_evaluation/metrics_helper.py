# =====================================================================
# metrics_helper.py - A scitoolkit file
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
from six import iteritems

# sklearn classification, regression, multi-label, clustering,
# and biclustering metrics
from sklearn import metrics as sk_metrics
# sklearn pairwise metrics
from sklearn.metrics import pairwise as sk_pairwise
# sklearn joblib for parallel computing
from sklearn.externals.joblib import delayed, Parallel
# sklearn utils
from sklearn.utils import gen_even_slices
# count of CPUs
from multiprocessing import cpu_count  # TODO use joblib?

import numpy as np  # NumPy
from functools import wraps

from scitoolkit.util.py_helper import is_str, reverse_dict, filter_unused_kwargs
from scitoolkit.model_evaluation.metrics import (
    mean_per_class_accuracy, root_mean_squared_error,
    normalized_root_mean_squared_error, mean_absolute_percent_error
)

# Classification
_METRICS_SCALAR_CLASSIFICATION = {
    'accuracy': sk_metrics.accuracy_score,
    'auc': sk_metrics.auc,
    'average_precision': sk_metrics.average_precision_score,
    'brier': sk_metrics.brier_score_loss,
    'cohen_kappa': sk_metrics.cohen_kappa_score,
    'f1': sk_metrics.f1_score,
    'fbeta': sk_metrics.fbeta_score,
    'hamming': sk_metrics.hamming_loss,
    'hinge': sk_metrics.hinge_loss,
    'jaccard_similarity': sk_metrics.jaccard_similarity_score,
    'log': sk_metrics.log_loss,
    'matthews': sk_metrics.matthews_corrcoef,
    'precision': sk_metrics.precision_score,
    'recall': sk_metrics.recall_score,
    'roc_auc': sk_metrics.roc_auc_score,
    'zero_one': sk_metrics.zero_one_loss,
    # non-sk
    'mean_per_class_accuracy': mean_per_class_accuracy,
}

_METRICS_MISC_CLASSIFICATION = {
    'classification_report': sk_metrics.classification_report,
    'confusion_matrix': sk_metrics.confusion_matrix,
    'precision_recall_curve': sk_metrics.precision_recall_curve,  # \/
    'pr_curve': sk_metrics.precision_recall_curve,  # /\
    'precision_recall_fscore_support':
        sk_metrics.precision_recall_fscore_support,
    'roc_curve': sk_metrics.roc_curve,
}

_METRICS_CLASSIFICATION = _METRICS_SCALAR_CLASSIFICATION.copy()
_METRICS_CLASSIFICATION.update(_METRICS_MISC_CLASSIFICATION)

# Regression
_METRICS_SCALAR_REGRESSION = {
    'explained_variance': sk_metrics.explained_variance_score,
    'mean_absolute_error': sk_metrics.mean_absolute_error,
    'mean_squared_error': sk_metrics.mean_squared_error,
    'mean_squared_log_error': sk_metrics.mean_squared_log_error,
    'median_absolute_error': sk_metrics.median_absolute_error,
    'r2': sk_metrics.r2_score,
    # non-sk
    'root_mean_squared_error': root_mean_squared_error,
    'normalized_root_mean_squared_error': normalized_root_mean_squared_error,
    'mean_absolute_percent_error': mean_absolute_percent_error
}

_METRICS_MISC_REGRESSION = {}

_METRICS_REGRESSION = _METRICS_SCALAR_REGRESSION.copy()
_METRICS_REGRESSION.update(_METRICS_MISC_REGRESSION)

# Multilabel
_METRICS_SCALAR_MULTILABEL = {
    'coverage_error': sk_metrics.coverage_error,
    'label_ranking_average_precision':
        sk_metrics.label_ranking_average_precision_score,
    'label_ranking_loss': sk_metrics.label_ranking_loss,
}
_METRICS_MISC_MULTILABEL = {}

_METRICS_MULTILABEL = _METRICS_SCALAR_MULTILABEL.copy()
_METRICS_MULTILABEL.update(_METRICS_MISC_MULTILABEL)

# Supervised (combined)
_METRICS_SCALAR_SUPERVISED = _METRICS_SCALAR_CLASSIFICATION.copy()
_METRICS_SCALAR_SUPERVISED.update(_METRICS_SCALAR_REGRESSION)
_METRICS_SCALAR_SUPERVISED.update(_METRICS_SCALAR_MULTILABEL)

_METRICS_MISC_SUPERVISED = _METRICS_MISC_CLASSIFICATION.copy()
_METRICS_MISC_SUPERVISED.update(_METRICS_MISC_REGRESSION)
_METRICS_MISC_SUPERVISED.update(_METRICS_MISC_MULTILABEL)

_METRICS_SUPERVISED = _METRICS_CLASSIFICATION.copy()
_METRICS_SUPERVISED.update(_METRICS_REGRESSION)
_METRICS_SUPERVISED.update(_METRICS_MULTILABEL)

# Clustering
_METRICS_SCALAR_CLUSTERING = {
    'adjusted_mutual_info': sk_metrics.adjusted_mutual_info_score,
    'adjusted_rand': sk_metrics.adjusted_rand_score,
    'calinski_harabaz': sk_metrics.calinski_harabaz_score,
    'completeness': sk_metrics.completeness_score,
    'fowlkes_mallows': sk_metrics.fowlkes_mallows_score,
    'homogeneity': sk_metrics.homogeneity_score,
    'mutual_info': sk_metrics.mutual_info_score,
    'normalized_mutual_info': sk_metrics.normalized_mutual_info_score,
    'silhouette': sk_metrics.silhouette_score,
    'v_measure': sk_metrics.v_measure_score,
}

_METRICS_MISC_CLUSTERING = {
    'silhouette_samples': sk_metrics.silhouette_samples,
    'homogeneity_completeness_v_measure':
        sk_metrics.homogeneity_completeness_v_measure,
}

_METRICS_CLUSTERING = _METRICS_SCALAR_CLUSTERING.copy()
_METRICS_CLUSTERING.update(_METRICS_MISC_CLUSTERING)

# Biclustering
_METRICS_SCALAR_BICLUSTERING = {
    'consensus': sk_metrics.consensus_score
}

_METRICS_MISC_BICLUSTERING = {}

_METRICS_BICLUSTERING = _METRICS_SCALAR_BICLUSTERING.copy()
_METRICS_BICLUSTERING.update(_METRICS_MISC_BICLUSTERING)

# Unsupervised (combine)
_METRICS_SCALAR_UNSUPERVISED = _METRICS_SCALAR_CLUSTERING.copy()
_METRICS_SCALAR_UNSUPERVISED.update(_METRICS_SCALAR_BICLUSTERING)

_METRICS_MISC_UNSUPERVISED = _METRICS_MISC_CLUSTERING.copy()
_METRICS_MISC_UNSUPERVISED.update(_METRICS_MISC_BICLUSTERING)

_METRICS_UNSUPERVISED = _METRICS_SCALAR_UNSUPERVISED.copy()
_METRICS_UNSUPERVISED.update(_METRICS_MISC_UNSUPERVISED)

# Pairwise
_METRICS_SCALAR_PAIRWISE = {}

_METRICS_MISC_PAIRWISE = {}
# Update with dict of kernel names and functions.
# >>> kernel_metrics()
# {'additive_chi2': sklearn.metrics.pairwise.additive_chi2_kernel,
#  'chi2': sklearn.metrics.pairwise.chi2_kernel,
#  'linear': sklearn.metrics.pairwise.linear_kernel,
#  'polynomial': sklearn.metrics.pairwise.polynomial_kernel,
#  'poly': sklearn.metrics.pairwise.polynomial_kernel,
#  'rbf': sklearn.metrics.pairwise.rbf_kernel,
#  'laplacian': sklearn.metrics.pairwise.laplacian_kernel,
#  'sigmoid': sklearn.metrics.pairwise.sigmoid_kernel,
#  'cosine': sklearn.metrics.pairwise.cosine_similarity}
# (Last Updated: sklearn.__version__ == 0.19.1)
_METRICS_MISC_PAIRWISE.update(sk_pairwise.kernel_metrics())
# Update with dict of distance names and functions.
# >>> distance_metrics()
# {'cityblock': sklearn.metrics.pairwise.manhattan_distances,  # \/
#  'cosine': sklearn.metrics.pairwise.cosine_distances,
#  'euclidean': sklearn.metrics.pairwise.euclidean_distances,  # \/
#  'l2': sklearn.metrics.pairwise.euclidean_distances,  # /\
#  'l1': sklearn.metrics.pairwise.manhattan_distances,  # \/
#  'manhattan': sklearn.metrics.pairwise.manhattan_distances,  # /\
#  'precomputed': None}
# (Last Updated: sklearn.__version__ == 0.19.1)
_METRICS_MISC_PAIRWISE.update(sk_pairwise.distance_metrics())
# Update with paired distance names (prepend "paired_") and functions.
# >>> {'paired_' + k: v for k, v in
# ...  iteritems(sk_pairwise.PAIRED_DISTANCES.copy())}
# {'paired_cosine': sklearn.metrics.pairwise.paired_cosine_distances,
#  'paired_euclidean': sklearn.metrics.pairwise.paired_euclidean_distances,
#  'paired_l2': sklearn.metrics.pairwise.paired_euclidean_distances,
#  'paired_l1': sklearn.metrics.pairwise.paired_manhattan_distances,
#  'paired_manhattan': sklearn.metrics.pairwise.paired_manhattan_distances,
#  'paired_cityblock': sklearn.metrics.pairwise.paired_manhattan_distances}
# (Last Updated: sklearn.__version__ == 0.19.1)
_METRICS_MISC_PAIRWISE.update(
    {'paired_' + k: v for k, v in
     iteritems(sk_pairwise.PAIRED_DISTANCES.copy())}
)

_METRICS_PAIRWISE = _METRICS_SCALAR_PAIRWISE.copy()
_METRICS_PAIRWISE.update(_METRICS_MISC_PAIRWISE)

# Combine scalar
_METRICS_SCALAR = _METRICS_SCALAR_SUPERVISED.copy()
_METRICS_SCALAR.update(_METRICS_SCALAR_UNSUPERVISED)
_METRICS_SCALAR.update(_METRICS_SCALAR_PAIRWISE)

# Combine misc.
_METRICS_MISC = _METRICS_MISC_SUPERVISED.copy()
_METRICS_MISC.update(_METRICS_MISC_UNSUPERVISED)
_METRICS_MISC.update(_METRICS_MISC_PAIRWISE)

# All combined
_METRICS = _METRICS_SCALAR.copy()
_METRICS.update(_METRICS_MISC)

# Reversed for name lookup
_METRICS_NAME_LOOKUP = reverse_dict(_METRICS)


def _get_metric_wrapper(s):
    def _get_metric_decorator(func):
        @wraps(func)
        def _call(name, *args, **kwargs):
            try:
                return func(name, *args, **kwargs)
            except KeyError:
                raise ValueError('Unknown {}: {}'.format(s, name))

        return _call

    return _get_metric_decorator


@_get_metric_wrapper('metric')
def _get_metric_func_by_name(name):
    return _METRICS[name.lower()]


@_get_metric_wrapper('scalar metric')
def _get_scalar_metric_func_by_name(name):
    return _METRICS_SCALAR[name.lower()]


@_get_metric_wrapper('misc metric')
def _get_misc_metric_func_by_name(name):
    return _METRICS_MISC[name.lower()]


@_get_metric_wrapper('supervised metric')
def _get_supervised_metric_func_by_name(name):
    return _METRICS_SUPERVISED[name.lower()]


@_get_metric_wrapper('supervised scalar metric')
def _get_supervised_scalar_metric_func_by_name(name):
    return _METRICS_SCALAR_SUPERVISED[name.lower()]


@_get_metric_wrapper('supervised misc metric')
def _get_supervised_misc_metric_func_by_name(name):
    return _METRICS_MISC_SUPERVISED[name.lower()]


@_get_metric_wrapper('unsupervised metric')
def _get_unsupervised_metric_func_by_name(name):
    return _METRICS_UNSUPERVISED[name.lower()]


@_get_metric_wrapper('unsupervised scalar metric')
def _get_unsupervised_scalar_metric_func_by_name(name):
    return _METRICS_SCALAR_UNSUPERVISED[name.lower()]


@_get_metric_wrapper('unsupervised misc metric')
def _get_unsupervised_misc_metric_func_by_name(name):
    return _METRICS_MISC_UNSUPERVISED[name.lower()]


@_get_metric_wrapper('pairwise metric')
def _get_pairwise_metric_func_by_name(name):
    return _METRICS_PAIRWISE[name.lower()]


@_get_metric_wrapper('pairwise scalar metric')
def _get_pairwise_scalar_metric_func_by_name(name):
    return _METRICS_SCALAR_PAIRWISE[name.lower()]


@_get_metric_wrapper('pairwise misc metric')
def _get_pairwise_misc_metric_func_by_name(name):
    return _METRICS_MISC_PAIRWISE[name.lower()]


@_get_metric_wrapper('clustering metric')
def _get_clustering_metric_func_by_name(name):
    return _METRICS_CLUSTERING[name.lower()]


@_get_metric_wrapper('clustering scalar metric')
def _get_clustering_scalar_metric_func_by_name(name):
    return _METRICS_SCALAR_CLUSTERING[name.lower()]


@_get_metric_wrapper('clustering misc metric')
def _get_clustering_misc_metric_func_by_name(name):
    return _METRICS_MISC_CLUSTERING[name.lower()]


@_get_metric_wrapper('biclustering metric')
def _get_biclustering_metric_func_by_name(name):
    return _METRICS_BICLUSTERING[name.lower()]


@_get_metric_wrapper('biclustering scalar metric')
def _get_biclustering_scalar_metric_func_by_name(name):
    return _METRICS_SCALAR_BICLUSTERING[name.lower()]


@_get_metric_wrapper('biclustering misc metric')
def _get_biclustering_misc_metric_func_by_name(name):
    return _METRICS_MISC_BICLUSTERING[name.lower()]


@_get_metric_wrapper('classification metric')
def _get_classification_metric_func_by_name(name):
    return _METRICS_CLASSIFICATION[name.lower()]


@_get_metric_wrapper('classification scalar metric')
def _get_classification_scalar_metric_func_by_name(name):
    return _METRICS_SCALAR_CLASSIFICATION[name.lower()]


@_get_metric_wrapper('classification misc metric')
def _get_classification_misc_metric_func_by_name(name):
    return _METRICS_MISC_CLASSIFICATION[name.lower()]


@_get_metric_wrapper('regression metric')
def _get_regression_metric_func_by_name(name):
    return _METRICS_REGRESSION[name.lower()]


@_get_metric_wrapper('regression scalar metric')
def _get_regression_scalar_metric_func_by_name(name):
    return _METRICS_SCALAR_REGRESSION[name.lower()]


@_get_metric_wrapper('regression misc metric')
def _get_regression_misc_metric_func_by_name(name):
    return _METRICS_MISC_REGRESSION[name.lower()]


@_get_metric_wrapper('multilabel metric')
def _get_multilabel_metric_func_by_name(name):
    return _METRICS_MULTILABEL[name.lower()]


@_get_metric_wrapper('multilabel scalar metric')
def _get_multilabel_scalar_metric_func_by_name(name):
    return _METRICS_SCALAR_MULTILABEL[name.lower()]


@_get_metric_wrapper('multilabel misc metric')
def _get_multilabel_misc_metric_func_by_name(name):
    return _METRICS_MISC_MULTILABEL[name.lower()]


def _parallel_pairwise(X, Y, func, n_jobs, **kwargs):
    """Break the pairwise matrix in n_jobs even slices
    and compute them in parallel
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/pairwise.py
    """
    if n_jobs < 0:
        n_jobs = max(cpu_count() + 1 + n_jobs, 1)

    if Y is None:
        Y = X

    if n_jobs == 1:
        return func(X, Y, **kwargs)

    fd = delayed(func)
    ret = Parallel(n_jobs=n_jobs, verbose=0)(
        fd(X, Y[s], **kwargs)
        for s in gen_even_slices(Y.shape[0], n_jobs))

    return np.hstack(ret)


# TODO: efficient distance (pairwise_distances)
# TODO: efficient argmin (pairwise_distances_argmin)
# TODO: efficient min (pairwise_distances_argmin_min)
#       == pairwise_distances(X, Y=Y, metric=metric).argmin(axis=axis)
#       == pairwise_distances(X, Y=Y, metric=metric).min(axis=axis))


def pairwise_distances_argmin_min(X, Y, axis=1, metric="euclidean",
                                  batch_size=500, metric_kwargs=None):
    """Compute minimum distances between one point and a set of points.
    This function computes for each row in X, the index of the row of Y which
    is closest (according to the specified distance). The minimal distances are
    also returned.
    This is mostly equivalent to calling:
        (pairwise_distances(X, Y=Y, metric=metric).argmin(axis=axis),
         pairwise_distances(X, Y=Y, metric=metric).min(axis=axis))
    but uses much less memory, and is faster for large arrays.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples1, n_features)
        Array containing points.
    Y : {array-like, sparse matrix}, shape (n_samples2, n_features)
        Arrays containing points.
    axis : int, optional, default 1
        Axis along which the argmin and distances are to be computed.
    metric : string or callable, default 'euclidean'
        metric to use for distance computation. Any metric from scitoolkit.cikit-learn
        or scipy.spatial.distance can be used.
        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.
        Distance matrices are not supported.
        Valid values for metric are:
        - from scitoolkit.cikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']
        - from scitoolkit.cipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
          'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
          'yule']
        See the documentation for scipy.spatial.distance for details on these
        metrics.
    batch_size : integer
        To reduce memory consumption over the naive solution, data are
        processed in batches, comprising batch_size rows of X and
        batch_size rows of Y. The default value is quite conservative, but
        can be changed for fine-tuning. The larger the number, the larger the
        memory usage.
    metric_kwargs : dict, optional
        Keyword arguments to pass to specified metric function.
    Returns
    -------
    argmin : numpy.ndarray
        Y[argmin[i], :] is the row in Y that is closest to X[i, :].
    distances : numpy.ndarray
        distances[i] is the distance between the i-th row in X and the
        argmin[i]-th row in Y.
    See also
    --------
    sklearn.metrics.pairwise_distances
    sklearn.metrics.pairwise_distances_argmin
    """
    dist_func = None
    if metric in PAIRWISE_DISTANCE_FUNCTIONS:
        dist_func = PAIRWISE_DISTANCE_FUNCTIONS[metric]
    elif not callable(metric) and not isinstance(metric, str):
        raise ValueError("'metric' must be a string or a callable")

    X, Y = check_pairwise_arrays(X, Y)

    if metric_kwargs is None:
        metric_kwargs = {}

    if axis == 0:
        X, Y = Y, X

    # Allocate output arrays
    indices = np.empty(X.shape[0], dtype=np.intp)
    values = np.empty(X.shape[0])
    values.fill(np.infty)

    for chunk_x in gen_batches(X.shape[0], batch_size):
        X_chunk = X[chunk_x, :]

        for chunk_y in gen_batches(Y.shape[0], batch_size):
            Y_chunk = Y[chunk_y, :]

            if dist_func is not None:
                if metric == 'euclidean':  # special case, for speed
                    d_chunk = safe_sparse_dot(X_chunk, Y_chunk.T,
                                              dense_output=True)
                    d_chunk *= -2
                    d_chunk += row_norms(X_chunk, squared=True)[:, np.newaxis]
                    d_chunk += row_norms(Y_chunk, squared=True)[np.newaxis, :]
                    np.maximum(d_chunk, 0, d_chunk)
                else:
                    d_chunk = dist_func(X_chunk, Y_chunk, **metric_kwargs)
            else:
                d_chunk = pairwise_distances(X_chunk, Y_chunk,
                                             metric=metric, **metric_kwargs)

            # Update indices and minimum values using chunk
            min_indices = d_chunk.argmin(axis=1)
            min_values = d_chunk[np.arange(chunk_x.stop - chunk_x.start),
                                 min_indices]

            flags = values[chunk_x] > min_values
            indices[chunk_x][flags] = min_indices[flags] + chunk_y.start
            values[chunk_x][flags] = min_values[flags]

    if metric == "euclidean" and not metric_kwargs.get("squared", False):
        np.sqrt(values, values)
    return indices, values


def pairwise_distances_argmin(X, Y, axis=1, metric="euclidean",
                              batch_size=500, metric_kwargs=None):
    """Compute minimum distances between one point and a set of points.
    This function computes for each row in X, the index of the row of Y which
    is closest (according to the specified distance).
    This is mostly equivalent to calling:
        pairwise_distances(X, Y=Y, metric=metric).argmin(axis=axis)
    but uses much less memory, and is faster for large arrays.
    This function works with dense 2D arrays only.
    Parameters
    ----------
    X : array-like
        Arrays containing points. Respective shapes (n_samples1, n_features)
        and (n_samples2, n_features)
    Y : array-like
        Arrays containing points. Respective shapes (n_samples1, n_features)
        and (n_samples2, n_features)
    axis : int, optional, default 1
        Axis along which the argmin and distances are to be computed.
    metric : string or callable
        metric to use for distance computation. Any metric from scitoolkit.cikit-learn
        or scipy.spatial.distance can be used.
        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.
        Distance matrices are not supported.
        Valid values for metric are:
        - from scitoolkit.cikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']
        - from scitoolkit.cipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
          'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
          'yule']
        See the documentation for scipy.spatial.distance for details on these
        metrics.
    batch_size : integer
        To reduce memory consumption over the naive solution, data are
        processed in batches, comprising batch_size rows of X and
        batch_size rows of Y. The default value is quite conservative, but
        can be changed for fine-tuning. The larger the number, the larger the
        memory usage.
    metric_kwargs : dict
        keyword arguments to pass to specified metric function.
    Returns
    -------
    argmin : numpy.ndarray
        Y[argmin[i], :] is the row in Y that is closest to X[i, :].
    See also
    --------
    sklearn.metrics.pairwise_distances
    sklearn.metrics.pairwise_distances_argmin_min
    """
    if metric_kwargs is None:
        metric_kwargs = {}

    return pairwise_distances_argmin_min(X, Y, axis, metric, batch_size,
                                         metric_kwargs)[0]


def _get_metrics_smart():
    # TODO: see http://www.chioka.in/differences-between-roc-auc-and-pr-auc/
    # Input to function is what parts of CM are important...
    pass


def _get_metric_name_by_func(func):
    return _METRICS_NAME_LOOKUP.get(func, str(func))


def eval_metrics(metric_names_or_funcs, *args,
                 remove_unused_kwargs=False, **kwargs):
    results = {}

    for metric in metric_names_or_funcs:
        metric_value = eval_metric(metric, *args,
                                   remove_unused_kwargs=remove_unused_kwargs,
                                   **kwargs)
        if not is_str(metric):
            metric = _get_metric_name_by_func(metric)
        results[metric] = metric_value

    return results


def eval_metric(metric_name_or_func, *args,
                remove_unused_kwargs=False, **kwargs):
    metric = metric_name_or_func

    if is_str(metric):
        metric = _get_metric_func_by_name(metric)
    if callable(metric):
        if remove_unused_kwargs:
            kwargs = filter_unused_kwargs(metric, kwargs)
        metric_value = metric(*args, **kwargs)
        result = metric_value
    else:
        raise ValueError('Unknown metric: {}'.format(metric))

    return result
