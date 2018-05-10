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
from scitoolkit.util.py23 import *

from deap import base, creator, tools
from scitoolkit.model_search.base import ModelSearchBase


def _init_individual():
    pass


class GeneticAlgorithm(ModelSearchBase):
    """
    Attributes:
        fitness_name
        fitness_weights
        See ModelSearchBase for others
    """

    def __init__(self, *args, population_size=50,
                 gene_mutation_prob=.1, gene_crossover_prob=.5,
                 tournament_size=3, num_generations=10,
                 n_jobs=1, score_on_err='raise', **kwargs):
        super(GeneticAlgorithm, self).__init__(*args, **kwargs)
        # Assign appropriate fitness attributes
        if self.maximize:  # TODO multi-output
            self.fitness_name = 'FitnessMax'
            self.fitness_weights = (1.0,)
        else:
            self.fitness_name = 'FitnessMin'
            self.fitness_weights = (-1.0,)
        # Define fitness type and individual
        creator.create(self.fitness_name, base.Fitness,
                       weights=self.fitness_weights)
        creator.create('Individual', list, est=self.model,
                       fitness=getattr(creator, self.fitness_name))
        # Initialize population
        self._toolbox = base.Toolbox()
        self._toolbox.register('individual', _init_individual,
                               creator.Individual)
        self._toolbox.register('population', tools.initRepeat, list,
                               self._toolbox.individual)
        if self.n_jobs != 1:
            self._toolbox.register('map', self.n_jobs)

    def search(self):
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
