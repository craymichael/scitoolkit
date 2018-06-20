# =====================================================================
# genetic_test.py - A scitoolkit file
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

from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris

from scitoolkit.model_search.genetic import GeneticAlgorithm
from scitoolkit.infrastructure import test


class GeneticAlgorithmTestCase(test.TestCase):

    def test_integration_svm(self):
        iris = load_iris()
        X, y = iris.data, iris.target

        hparam_space = {}

        ga = GeneticAlgorithm(model=LinearSVC, hparam_space=hparam_space,
                              n_jobs=1, iid=True, maximize=True,
                              ckpt_every=None, dirname=None, basename=None,
                              keep_recent=5, verbose=1, metrics=None,
                              target_metric=None, classification=True)

        ga.search(X, y)
