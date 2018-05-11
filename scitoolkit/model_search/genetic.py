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

from deap import base, creator, tools, algorithms
import numpy as np

from scitoolkit.model_search.base import ModelSearchBase
from scitoolkit.util.parallel import map


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

        self.num_generations = num_generations
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
        # History and logs
        self.all_history = []
        self.all_logbooks = []
        # Placeholders
        self._toolbox = None
        self._pop = None
        self._hof = None
        self._stats = None
        self._hist = None

    def _init_toolbox(self):
        # Initialize population
        self._toolbox = base.Toolbox()
        self._toolbox.register('individual', _init_individual,
                               creator.Individual)
        self._toolbox.register('population', tools.initRepeat, list,
                               self._toolbox.individual)
        if self.n_jobs != 1:
            def _map_func(*_args, **_kwargs):
                return map(*_args, n_jobs=self.n_jobs, verbose=self.verbose,
                           **_kwargs)

            self._toolbox.register('map', _map_func)

        self._toolbox.register('evaluate', _evalFunction,
                               name_values=name_values, X=X, y=y,
                               scorer=self.scorer_, cv=cv, iid=self.iid,
                               verbose=self.verbose, error_score=self.error_score,
                               fit_params=self.fit_params, score_cache=self.score_cache)

        self._toolbox.register('mate', _cxIndividual, indpb=self.gene_crossover_prob,
                               gene_type=self.gene_type)

        self._toolbox.register('mutate', _mutIndividual, indpb=self.gene_mutation_prob,
                               up=maxints)
        self._toolbox.register('select', tools.selTournament,
                               tournsize=self.tournament_size)

        self._pop = self._toolbox.population(n=self.population_size)
        self._hof = tools.HallOfFame(1)

        # Stats
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('avg', np.nanmean)
        stats.register('min', np.nanmin)
        stats.register('max', np.nanmax)
        stats.register('std', np.nanstd)
        self._stats = stats

        # History
        hist = tools.History()
        self._toolbox.decorate('mate', hist.decorator)
        self._toolbox.decorate('mutate', hist.decorator)
        hist.update(self._pop)
        self._hist = hist

    def search(self):
        self._init_toolbox()

        if self.verbose:
            print('--- Evolve in {0} possible combinations ---'.format(
                np.prod(np.array(maxints) + 1)))

        # TODO: probs....
        # TODO: copy source for this func so can perform checkpointing (see DEAP docs)
        pop, logbook = algorithms.eaSimple(self._pop, self._toolbox, cxpb=0.5, mutpb=0.2,
                                           ngen=self.num_generations, stats=self._stats,
                                           halloffame=self._hof, verbose=self.verbose)

        # Save History
        self.all_history.append(self._hist)
        self.all_logbooks.append(logbook)
        current_best_score_ = self._hof[0].fitness.values[0]
        current_best_params_ = _individual_to_params(self._hof[0], name_values)
        if self.verbose:
            print('Best individual is: %s\nwith fitness: %s' % (
                current_best_params_, current_best_score_))

        if current_best_score_ > self.best_mem_score_:  # TODO
            self.best_mem_score_ = current_best_score_
            self.best_mem_params_ = current_best_params_

        self.best_score_ = current_best_score_
        self.best_params_ = current_best_params_

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
