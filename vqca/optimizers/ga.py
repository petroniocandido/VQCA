from deap import base
from deap import creator
from deap import tools
import numpy as np
import random

from .base import VQCAOptimizer

class VQCAOptimizerGA(VQCAOptimizer):
  name = 'GA'
  def __init__(self, **kwargs):
    super(VQCAOptimizerGA, self).__init__(**kwargs)
    self.no_of_generations = kwargs.get('no_of_generations',20)
    self.population_size = kwargs.get('population_size',30)
    self.probability_of_mutation = 0.3

  def log(self, loss, param):
    self.loss_history.append(loss)
    self.param_history.append(list(param))
    if np.min(loss) < self.min_loss:
      self.min_loss = np.min(loss)
      self.best_param = list(param)
      print("NEW MIN: {}\t{}".format(self.min_loss, self.best_param))

  def funcao_custo(self, parametros):
    return [super().funcao_custo(parametros)]

  def training_loop(self, param = None):

    objetivo = lambda x: self.funcao_custo(x)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    # an Individual is a list with one more attribute called fitness
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    
    if param is None:

      toolbox.register("attr_float", random.uniform, 0, 2*np.pi)
      toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, self.num_param)
      toolbox.register("population", tools.initRepeat, list, toolbox.individual)
      pop = toolbox.population(n=self.population_size)

    else: 
      pop = [creator.Individual(ind) for ind in param]

    toolbox.register("evaluate", objetivo) # privide the objective function here

    # registering basic processes using bulit in functions in DEAP
    toolbox.register("mate", tools.cxBlend, alpha=0.7) # strategy for crossover
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=self.probability_of_mutation) # mutation strategy with probability of mutation
    toolbox.register("select", tools.selBest) # selection startegy

    # Elitism
    hall_of_fame = tools.HallOfFame(1)

    

    # The next thing to do is to evaluate our brand new population.

    # use map() from python to give each individual to evaluate and create a list of the result
    fitnesses = list(map(toolbox.evaluate, pop))

    best_fit = np.inf
    best_ind = None

    # ind has individual and fit has fitness score
    for ind, fit in zip(pop, fitnesses):
      ind.fitness.values = fit
      if fit[0] < best_fit:
        best_fit = fit[0]
        best_ind = list(ind)

    # evolve our population until we reach the number of generations

    g = 0
    hall_of_fame.clear()

    patience = 10
    patience_count = 0

    # Begin the evolution
    while g < self.no_of_generations and patience_count <= patience:
        # A new generation
        g = g + 1

        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals, this needs to be done to create copy and avoid problem of inplace operations
        # This is of utter importance since the genetic operators in toolbox will modify the provided objects in-place.
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
          if random.random() < 0.5:
            toolbox.mate(child1, child2)
            child1.fitness.values = toolbox.evaluate(child1)
            child2.fitness.values = toolbox.evaluate(child2)


        for mutant in offspring:
          if random.random() < 0.3:
            toolbox.mutate(mutant)
            mutant.fitness.values = toolbox.evaluate(mutant)

        this_gen_fitness = [] # this list will have fitness value of all the offspring
        improved = False
        for ind in offspring:
            this_gen_fitness.append(ind.fitness.values[0])
            if ind.fitness.values[0] < best_fit:
              best_fit = ind.fitness.values[0]
              best_ind = ind
              patience_count = 0
              improved = True

        if not improved:
          patience_count += 1

        hall_of_fame.update(offspring)

        stmin = np.min(this_gen_fitness)
        stmen = np.mean(this_gen_fitness)
        ststd = np.std(this_gen_fitness)
        stmax = np.max(this_gen_fitness)

        self.log([stmin, stmen, ststd, stmax], best_ind)

        pop[:] = offspring
