# =====================================================================
# base.py - A scitoolkit file
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

import numpy as np
import six
import abc
import os
from datetime import datetime
from collections import OrderedDict
from sklearn.externals.joblib import (Parallel, delayed, cpu_count, Memory,
                                      dump, load)
from scitoolkit.system.file_system import (join, get_most_recent_in_dir,
                                           get_most_recent_k_in_dir)

MODEL_DIR = 'models'
SEARCH_DIR = 'search'


class ParamSpace(object):
    """
    {
        'lr': dict(
            lower=1e-4,
            upper=1,
            n_points=50,
            spacing='linear'
        ),
        'reg': dict(
            lower=1e-4,
            upper=1,
            n_points=50,
            spacing='log',
            also=[0]
        ),
        'weight_init': dict(
            categories=[
                'uniform',
                'normal'
            ],
            conditional={
                'normal': ['sigma', 'mu']
            }
        ),
        'sigma': dict(
            lower=5e-2,
            upper=1,
            n_points=25,
            spacing='linear'
        ),
        'mu': ...
    }
    """

    def __init__(self, hparam_dict):
        hparam_space = OrderedDict()
        dep_graph = _Graph()

        for k, v in six.iteritems(hparam_dict):
            assert isinstance(v, dict)

            if 'categories' in v:
                hparam_space[k] = _CategoricalParam(**v)
            else:
                hparam_space[k] = _ContinuousParam(**v)

            if hparam_space[k].conditional:
                # TODO this won't work for deep dependencies...
                for cond_k, cond_vs in six.iteritems(
                        hparam_space[k].conditional):
                    if k in dep_graph:
                        node = dep_graph[k]
                    else:
                        node = _Node()
                        dep_graph[k] = node
                    for cond_v in cond_vs:
                        node_v = _Node(cond_k)
                        dep_graph[k + '__' + cond_v] = node_v
                        node_v.add_parent(node)
                        node.add_child(node_v)

        self.hparam_space = hparam_space
        self.dep_graph = dep_graph

    def __iter__(self):
        return six.itervalues(self.hparam_space)

    def __len__(self):
        # The number of hyperparameters
        return len(self.hparam_space)

    def get_hparams(self, indices):
        assert len(indices) == len(self)

        invalid = []
        hparams = {}
        for idx, (k, hp) in zip(indices, six.iteritems(self.hparam_space)):
            hparams[k] = hp[idx]
            if hp[idx] in hp.conditional:  # TODO this is wrong and incomplete
                # Record hyperparameters
                invalid.extend(hp.conditional[hp[idx]])
        return hparams


class _Graph(object):
    def __init__(self):
        self.str_to_node = {}

    def __getitem__(self, item):
        return self.str_to_node[item]

    def __setitem__(self, key, value):
        self.str_to_node[key] = value

    def __contains__(self, item):
        return item in self.str_to_node


class _Node(object):
    def __init__(self, value=None):
        self.children = []
        self.parents = []
        # Property:
        self.value = value

    def add_parent(self, *parent):
        self.parents.extend(parent)

    def add_child(self, *child):
        self.children.extend(child)


class _Param(six.with_metaclass(abc.ABCMeta, object)):
    """"""
    def __init__(self, conditional):
        self.values = None
        self.conditional = conditional or {}

    @abc.abstractmethod
    @property
    def type(self):
        pass

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        return self.values[item]


class _ContinuousParam(_Param):
    """"""

    def __init__(self, lower, upper, n_points, spacing='linear', also=None,
                 conditional=None):
        super(_ContinuousParam, self).__init__(conditional)
        self.lower = lower
        self.upper = upper
        self.spacing = spacing  # linear, log
        if self.spacing == 'linear':
            self.values = np.linspace(lower, upper, num=n_points, endpoint=True)
        elif self.spacing == 'log':
            self.values = np.logspace(lower, upper, num=n_points, endpoint=True)
        else:
            raise ValueError('The spacing "{}" is '
                             'invalid.'.format(self.spacing))
        if also is not None:
            self.values = np.concatenate([also, self.values])
            # Sort values as order matters (low to high...)
            self.values = np.sort(self.values)

    @property
    def type(self):
        return 'continuous'


class _CategoricalParam(_Param):
    """"""

    def __init__(self, categories, conditional=None):
        super(_CategoricalParam, self).__init__(conditional)
        self.values = categories

    @property
    def type(self):
        return 'categorical'


class ModelSearchBase(six.with_metaclass(abc.ABCMeta, object)):
    """"""

    # TODO gen random states and save for split reproducability, etc.
    # TODO take in list of collections (pickleable)
    # TODO logging

    def __init__(self, model, hparam_space, cv=None, n_jobs=1, iid=True,
                 maximize=True, ckpt_every=None, dirname=None, basename=None,
                 keep_recent=5, verbose=0):
        # TODO move iid to scoring methodology? cv-only param I think...
        self.model = model
        self.hparam_space = hparam_space
        self.maximize = maximize
        self.cv = cv
        self.iid = iid
        self.best_model = None
        self.best_hparams = None
        self.best_score = None
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.ckpt_every = ckpt_every
        self.keep_recent = keep_recent
        # Create model directories and files
        if dirname is None:
            dirname = SEARCH_DIR
        if basename is None:
            basename = self.__class__.__name__
        self.dirname = dirname
        self.basename = basename

    def search(self):
        result = self._search()
        self.save()
        return result

    @abc.abstractmethod
    def _search(self):
        pass

    def save(self, filename=None, ext='pkl', timestamp=True, model_name=True):
        file_split = []

        if filename is not None:
            file_split.append(filename)

        if model_name:
            if file_split:
                prefix = '-'
            else:
                prefix = ''
            file_split.extend([prefix + 'model', self.model.__class__.__name__])

        if timestamp or not file_split:
            file_split.append(str(datetime.now()))

        filename = '_'.join(file_split)

        if ext is not None:
            filename = '.'.join([filename, ext])

        path = join(self.dirname, self.basename, filename)
        dump(self, filename=path)

        if self.keep_recent is not None:
            recent_save_fns, all_save_fns = get_most_recent_k_in_dir(
                os.path.dirname(path), k=self.keep_recent, delim='_', ext=ext,
                return_fns=True
            )

            if len(all_save_fns) > self.keep_recent:
                # Compute set of old save file names and delete those found
                goodbye = set(all_save_fns) - set(recent_save_fns)

                for save_fn in goodbye:
                    os.remove(save_fn)

    @staticmethod
    def load(filename, ext='pkl'):
        if os.path.isdir(filename):
            # Raises error if timestamped files aren't found (according to
            # dateutil)
            filename = get_most_recent_in_dir(filename, delim='_', ext=ext,
                                              raise_=True)
        return load(filename)
