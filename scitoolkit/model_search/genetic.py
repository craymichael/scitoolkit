# =====================================================================
# genetic.py - A scitoolkit file
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

from deap import base, creator


class GeneticAlgorithm(object):
    pass

# import sklearn.datasets
# import numpy as np
# import random
# data = sklearn.datasets.load_digits()
# X = data["data"]
# y = data["target"]
# from sklearn.svm import SVC
# from sklearn.model_selection import StratifiedKFold
# paramgrid = {"kernel": ["rbf"],
#              "C"     : np.logspace(-9, 9, num=25, base=10),
#              "gamma" : np.logspace(-9, 9, num=25, base=10)}
# random.seed(1)
# from evolutionary_search import EvolutionaryAlgorithmSearchCV
# cv = EvolutionaryAlgorithmSearchCV(estimator=SVC(),
#                                    params=paramgrid,
#                                    scoring="accuracy",
#                                    cv=StratifiedKFold(n_splits=10),
#                                    verbose=1,
#                                    population_size=50,
#                                    gene_mutation_prob=0.10,
#                                    gene_crossover_prob=0.5,
#                                    tournament_size=3,
#                                    generations_number=10)
# cv.fit(X, y)
