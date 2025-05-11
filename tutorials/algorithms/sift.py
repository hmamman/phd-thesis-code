
from tutorials.algorithms.phs import PHS


class SIFT(PHS):
    def __init__(self, mu, bounds, fitness_func):
        super().__init__(mu, bounds, fitness_func)

    def evaluate_fitness(self, offsprings):
        return self.fitness_func(offsprings)

