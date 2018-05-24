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
import random

import networkx
import matplotlib.pyplot as plt  # TODO plt...

from datetime import datetime  # TODO
import csv  # TODO
from scitoolkit.system.file_system import valid_filename  # TODO

from scitoolkit.model_search.base import ModelSearchBase
from scitoolkit.model_evaluation.eval import train_and_eval
from scitoolkit.util.parallel import map_jobs
from scitoolkit.util.py_helper import func_arg_names, get_default_args

DEFAULT_GA_PARAMS = dict(
    cxpb=0.5,
    mutpb=0.2,
    mu=50,
    lambda_=50
)


def _init_individual(cls, hparam_space):
    return cls(random.randint(0, len(hp) - 1) for hp in hparam_space)


def _mut_individual_grid(individual, hparam_space, indpb):
    # Perform for both categorical/continuous
    for i, hp in enumerate(hparam_space):
        if random.random() <= indpb:
            individual[i] = random.randint(0, len(hp) - 1)
    return individual,  # Tuple


def _cx_individual_grid(ind1, ind2, hparam_space, indpb):
    for i, hp in enumerate(hparam_space):
        if random.random() > indpb:
            continue
        if hp.type == 'categorical':
            ind1[i], ind2[i] = ind2[i], ind1[i]
        elif hp.type == 'continuous':
            # Case when parameters are numerical
            if ind1[i] <= ind2[i]:
                ind1[i] = random.randint(ind1[i], ind2[i])
                ind2[i] = random.randint(ind1[i], ind2[i])
            else:
                ind1[i] = random.randint(ind2[i], ind1[i])
                ind2[i] = random.randint(ind2[i], ind1[i])
        else:
            raise ValueError('Unknown HP type: {}'.format(hp.type))

    return ind1, ind2


def _individual_to_hparams_grid(individual, hparam_space):
    # Convert from indices to param dict to pass to model
    return hparam_space.get_hparams(individual)


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

    def __init__(self, *args, **kwargs):
        # GA params
        self.log = kwargs.get('log', True)  # TODO...
        self.population_size = kwargs.pop('population_size', 50)
        self.gene_mutation_prob = kwargs.pop('gene_mutation_prob', 0.1)
        self.gene_crossover_prob = kwargs.pop('gene_crossover_prob', 0.5)
        self.tournament_size = kwargs.pop('tournament_size', 3)
        self.num_generations = kwargs.pop('num_generations', 10)
        self.n_jobs = kwargs.pop('n_jobs', 1)
        self.grid = kwargs.pop('grid', True)
        if not self.grid:
            raise NotImplementedError
        # TODO i forgot what this is...
        self.score_on_err = kwargs.pop('score_on_err', 'raise')
        # Wrap ga_func
        ga_func = kwargs.pop('ga_func', eaSimple1Gen)
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

        super(GeneticAlgorithm, self).__init__(*args, **kwargs)

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
        creator.create('Individual', list,
                       fitness=getattr(creator, self.fitness_name))
        # Init toolbox
        self._init_toolbox()

    def _init_toolbox(self):
        # Initialize population
        self._toolbox = base.Toolbox()
        # noinspection PyUnresolvedReferences
        self._toolbox.register('individual', _init_individual,
                               creator.Individual,
                               hparam_space=self.hparam_space)
        # noinspection PyUnresolvedReferences
        self._toolbox.register('population', tools.initRepeat, list,
                               self._toolbox.individual)
        if self.n_jobs != 1:
            def _map_func(*_args, **_kwargs):
                return map_jobs(*_args, n_jobs=self.n_jobs,
                                verbose=self.verbose, **_kwargs)

            self._toolbox.register('map', _map_func)

        self._toolbox.register('mate', _cx_individual_grid,
                               indpb=self.gene_crossover_prob,
                               hparam_space=self.hparam_space)

        self._toolbox.register('mutate', _mut_individual_grid,
                               indpb=self.gene_mutation_prob,
                               hparam_space=self.hparam_space)

        self._toolbox.register('select', tools.selTournament,
                               tournsize=self.tournament_size)

        # noinspection PyUnresolvedReferences
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

    def _search(self, X, y):
        def _train_and_eval_wrapped(individual, *args, **kwargs):
            log = kwargs.pop('log')  # TODO
            hparams = _individual_to_hparams_grid(individual, self.hparam_space)
            model = self.model(**hparams)
            try:
                score, all_scores = train_and_eval(*args, model=model, **kwargs)
            # TODO score_on_err and shouldn't this be in base.py somewhere...
            except np.linalg.linalg.LinAlgError:  # TODO more exceptions...
                # TODO log exceptions...
                if self.maximize:
                    score = -np.inf
                else:
                    score = np.inf
                all_scores = {}

            if np.isnan(score):  # TODO
                if self.maximize:
                    score = -np.inf
                else:
                    score = np.inf

            if log is not None:
                row = [hparams[k]
                       for k in self.hparam_space.hparam_space.keys()]
                row.extend([all_scores.get(k) for k in self.metrics])
                row.append(score)
                row = list(map(str, row))
                log.writerow(row)
            return score,  # Tuple

        if self.log:
            # TODO fix/move
            f = open(valid_filename(self.__class__.__name__ +
                                    self.model.__name__ +
                                    str(datetime.now()) + '.csv'),
                     'w', buffering=1)  # buffering=1: buffer line by line
            writer = csv.writer(f)
            writer.writerow(list(self.hparam_space.hparam_space.keys()) +
                            self.metrics + [self.target_metric])
        else:
            f = None
            writer = None

        self._toolbox.register('evaluate', _train_and_eval_wrapped, X=X, y=y,
                               train_func='train', test_func='predict',
                               cv=self.cv, iid=self.iid,
                               return_train_score=False, metrics=self.metrics,
                               target_metric=self.target_metric, log=writer,
                               eval_kwargs={'throw_out': self.throw_out})  # TODO throw_out...

        if self.verbose:
            print('--- Evolve in {0} possible combinations ---'.format(
                np.prod(np.asarray([len(hp) for hp in self.hparam_space]))))

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
            # noinspection PyUnresolvedReferences
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
        best_params_ = _individual_to_hparams_grid(self._hof[0],
                                                   self.hparam_space)

        if self.verbose:
            print('Best individual is: {}\nwith fitness: {}'.format(
                best_params_, best_score_))

        if self.log:
            f.close()

        # TODO TODO
        # graph = networkx.DiGraph(self._hist.genealogy_tree)
        # graph = graph.reverse()  # Make the graph top-down
        # noinspection PyUnresolvedReferences
        # colors = [self._toolbox.evaluate(self._hist.genealogy_history[i])[0]
        #          for i in graph]
        # networkx.draw(graph, node_color=colors)
        # plt.savefig('gene_tree.png', dpi=300)
        # plt.show()
