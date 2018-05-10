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
from scitoolkit.py23 import *

import numpy as np
import six
from abc import ABCMeta
from joblib import Parallel, delayed, cpu_count, Memory, dump, load


class ParamSpace(object):
    """"""

    def __init__(self):
        pass
    # TODO categorical/continuous parameters


class ModelSearch(six.with_metaclass(ABCMeta, object)):
    """"""

    def __init__(self, model, param_space, population_size=50,
                 gene_mutation_prob=.1, gene_crossover_prob=.5,
                 tournament_size=3, num_generations=10,
                 n_jobs=1, score_on_err='raise', iid=True):
        pass

    def _build_model(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
