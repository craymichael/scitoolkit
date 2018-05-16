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
from scitoolkit.util.py_helper import func_arg_names, get_default_args

DEFAULT_GA_PARAMS = dict(
    cxpb=0.5,
    mutpb=0.2,
    mu=50,
    lambda_=50
)


def _init_individual():
    pass


def eaSimple1Gen(population, toolbox, cxpb, mutpb, gen):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param gen: The generation number.
    :returns: The offspring and population after 1 generation.

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution (if
    any). The logbook will contain the generation number, the number of
    evaluations for each generation and the statistics if a
    :class:`~deap.tools.Statistics` if any. The *cxpb* and *mutpb* arguments
    are passed to the :func:`varAnd` function. The pseudocode goes as follow
    ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    # 1 iteration of eaSimple (deap.__version__ == '1.0')
    if gen != 0:
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Replace the current population by the offspring
        population[:] = offspring
    else:
        # Generation 0
        offspring = population

    return offspring, population


def eaMuPlusLambda1Gen(population, toolbox, mu, lambda_, cxpb, mutpb, gen):
    """This is the :math:`(\mu + \lambda)` evolutionary algorithm.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param gen: The generation number.
    :returns: The offspring and population after 1 generation.

    The algorithm takes in a population and evolves it in place using the
    :meth:`varOr` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution (if
    any). The logbook will contain the generation number, the number of
    evaluations for each generation and the statistics if a
    :class:`~deap.tools.Statistics` if any. The *cxpb* and *mutpb* arguments
    are passed to the :func:`varAnd` function. The pseudocode goes as follow
    ::

        evaluate(population)
        for g in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(population + offspring, mu)

    First, the individuals having an invalid fitness are evaluated. Second,
    the evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by the :func:`varOr` function. The
    offspring are then evaluated and the next generation population is
    selected from both the offspring **and** the population. Finally, when
    *ngen* generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Care must be taken when the lambda:mu ratio is 1 to 1 as a non-stochastic
        selection will result in no selection at all as
        the operator selects *lambda* individuals from a pool of *mu*.

    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """
    # 1 iteration of eaMuPlusLambda (deap.__version__ == '1.0')
    if gen != 0:
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)
    else:
        # Generation 0
        offspring = population

    return offspring, population


def eaMuCommaLambda1Gen(population, toolbox, mu, lambda_, cxpb, mutpb, gen):
    """This is the :math:`(\mu~,~\lambda)` evolutionary algorithm.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param gen: The generation number.
    :returns: The offspring and population after 1 generation.

    First, the individuals having an invalid fitness are evaluated. Then, the
    evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by a crossover, a mutation or a
    reproduction proportionally to the probabilities *cxpb*, *mutpb* and 1 -
    (cxpb + mutpb). The offspring are then evaluated and the next generation
    population is selected **only** from the offspring. Briefly, the operators
    are applied as following ::

        evaluate(population)
        for i in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(offspring, mu)

    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """
    # 1 iteration of eaMuCommaLambda (deap.__version__ == '1.0')
    assert lambda_ >= mu, ('lambda ({}) must be greater or equal to mu '
                           '({}).'.format(lambda_, mu))

    if gen != 0:
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)
    else:
        # Generation 0
        offspring = population

    return offspring, population


class GeneticAlgorithm(ModelSearchBase):
    """
    Attributes:
        fitness_name
        fitness_weights
        gen
        start_gen
        See ModelSearchBase for others
    """

    def __init__(self, *args, population_size=50,
                 gene_mutation_prob=.1, gene_crossover_prob=.5,
                 tournament_size=3, num_generations=10,
                 n_jobs=1, score_on_err='raise', ga_func=eaSimple1Gen,
                 **kwargs):
        super(GeneticAlgorithm, self).__init__(*args, **kwargs)

        # GA params
        self.population_size = population_size
        self.num_generations = num_generations
        self.tournament_size = tournament_size
        self.n_jobs = n_jobs
        self.gene_mutation_prob = gene_mutation_prob
        self.gene_crossover_prob = gene_crossover_prob
        # Wrap ga_func
        ga_func_args = {}
        ga_func_defaults = get_default_args(ga_func)
        for arg_name in func_arg_names(ga_func):
            if arg_name not in {'population', 'toolbox', 'gen'}:
                if arg_name in kwargs:
                    arg_val = kwargs[arg_name]
                elif arg_name in ga_func_defaults:
                    # NOTE: If function provides default, DEFAULT_GA_PARAMS
                    #       is ignored
                    continue
                elif arg_name in DEFAULT_GA_PARAMS:
                    arg_val = DEFAULT_GA_PARAMS[arg_name]
                else:
                    raise ValueError('Specified `ga_func` {} requires arg '
                                     '`{}`.'.format(ga_func, arg_name))
                ga_func_args[arg_name] = arg_val

        def ga_func_wrapped(population, toolbox, gen):
            return ga_func(population, toolbox, gen=gen, **ga_func_args)

        self._ga_func = ga_func_wrapped
        self._score_on_err = score_on_err  # TODO i forgot what this is...
        # Init values
        self.start_gen = 0
        self.gen = 0  # Running var
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
        # Init toolbox
        self._init_toolbox()

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
        stats.register('median', np.nanmedian)
        self._stats = stats

        # History
        hist = tools.History()
        self._toolbox.decorate('mate', hist.decorator)
        self._toolbox.decorate('mutate', hist.decorator)
        hist.update(self._pop)
        self._hist = hist

    def _search(self):
        if self.verbose:
            print('--- Evolve in {0} possible combinations ---'.format(
                np.prod(np.array(maxints) + 1)))

        # Init logbook
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (self._stats.fields
                                              if self._stats else [])

        # Begin the generational process
        for gen in range(self.start_gen, self.num_generations + 1):
            self.gen = gen
            # Compute 1 generation of GA
            offspring, population = self._ga_func(self._pop, self._toolbox,
                                                  gen=self.gen)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self._toolbox.map(self._toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if self._hof is not None:
                self._hof.update(offspring)

            # Append the current generation statistics to the logbook
            record = (self._stats.compile(population)
                      if self._stats is not None else {})
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if self.verbose:
                print(logbook.stream)

            # Checkpoint
            if self.ckpt_every and (gen + 1) % self.ckpt_every == 0:
                self.save(filename='generation_{}'.format(self.gen),
                          ext='pkl', timestamp=True)

        best_score_ = self._hof[0].fitness.values[0]
        best_params_ = _individual_to_params(self._hof[0], name_values)
        if self.verbose:
            print('Best individual is: {}\nwith fitness: {}'.format(
                best_params_, best_score_))

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
