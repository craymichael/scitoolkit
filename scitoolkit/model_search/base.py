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
from sklearn.externals.joblib import Parallel, delayed, cpu_count, Memory, dump, load, pool


class ParamSpace(object):
    """"""

    def __init__(self):
        pass
    # TODO define categorical/continuous parameters


class ModelSearchBase(six.with_metaclass(abc.ABCMeta, object)):
    """"""
    # TODO gen random states and save for split reproducability, etc.
    # TODO take in list of collections (pickleable)

    def __init__(self, model, hparam_space, n_jobs=1, iid=True, maximize=True):
        # TODO move iid to scoring methodology? cv-only param I think...
        self.model = model
        self.maximize = maximize
        self.best_model = None
        self.best_hparams = None
        self.best_score = None
        self.n_jobs = n_jobs

    @abc.abstractmethod
    def _build_model(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
