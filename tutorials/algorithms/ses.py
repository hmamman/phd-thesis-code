import numpy as np


class SES:
    def __init__(self, mu, lambda_, sigma, bounds, fitness_func):
        self.population = []
        self.mu = mu
        self.lambda_ = lambda_
        self.sigma = sigma
        self.bounds = np.array(bounds)
        self.dimension = len(bounds)
        self.feature_ranges = self.bounds[:, 1] - self.bounds[:, 0]
        self.features_size = max(1, self.dimension)
        self.features_to_select = list(range(self.dimension))
        self.feature_to_mutate = list(range(self.dimension))

        self.fitness_func = fitness_func

        self.initialize_population()

    def initialize_population(self):
        # Generate the entire population randomly
        for _ in range(self.mu):
            solution = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            self.population.append(solution)

        self.population = np.array(self.population)

    def global_generation(self, base_solution):
        offspring = base_solution.copy()

        # Generate mutations for selected features all at once
        mutations = np.random.normal(0, self.sigma, size=len(self.feature_to_mutate)) * self.feature_ranges[self.feature_to_mutate]

        # Apply mutations vectorized
        offspring[self.feature_to_mutate] += mutations

        return np.clip(offspring, self.bounds[:, 0], self.bounds[:, 1])

    def generate_offsprings(self, base_solution):
        offsprings = []

        for _ in range(self.lambda_):
            offspring = self.global_generation(base_solution)
            offsprings.append(offspring)

        return offsprings

    def get_offsprings(self):
        all_offsprings = []
        for base_solution in self.population:
            offsprings = self.generate_offsprings(base_solution)
            all_offsprings.extend(offsprings)
        return np.array(all_offsprings)

    def evaluate_fitness(self, offsprings):
        fitness = np.array([self.fitness_func(c) for c in offsprings])
        return fitness

    # Apply rank selection
    def selection(self, offsprings, fitness):
        best_indices = np.argsort(fitness)[-self.mu:]  # Minimization
        parents = offsprings[best_indices]
        return parents

    def run(self):
        offsprings = self.get_offsprings()
        fitness = self.evaluate_fitness(offsprings)

        self.population = self.selection(offsprings, fitness)