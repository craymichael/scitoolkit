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
from sklearn.externals.joblib import (Parallel, delayed, cpu_count, Memory, dump,
                                      load)
from scitoolkit.system.file_system import (join, get_most_recent_in_dir,
                                           get_most_recent_k_in_dir)

MODEL_DIR = 'models'
SEARCH_DIR = 'search'


class ParamSpace(object):
    """"""

    def __init__(self):
        pass
    # TODO define categorical/continuous parameters


class ModelSearchBase(six.with_metaclass(abc.ABCMeta, object)):
    """"""

    # TODO gen random states and save for split reproducability, etc.
    # TODO take in list of collections (pickleable)

    def __init__(self, model, hparam_space, n_jobs=1, iid=True,
                 maximize=True, ckpt_every=None, dirname=None, basename=None,
                 keep_recent=5, verbose=0):
        # TODO move iid to scoring methodology? cv-only param I think...
        self.model = model
        self.maximize = maximize
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
                file_split.extend(['-model', self.model.__class__.__name__])
            else:
                file_split.extend(['model', self.model.__class__.__name__])

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
                return_fns=True)

            if len(all_save_fns) > self.keep_recent:
                # Compute set of old save file names
                goodbye = set(all_save_fns) - set(recent_save_fns)

                for save_fn in goodbye:
                    os.remove(save_fn)

    @staticmethod
    def load(filename, ext='pkl'):
        if os.path.isdir(filename):
            # Raises error if timestamped files aren't found (according to dateutil)
            filename = get_most_recent_in_dir(filename, delim='_', ext=ext,
                                              raise_=True)
        return load(filename)
