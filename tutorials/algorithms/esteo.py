import os
import sys

import numpy as np
# Get the absolute path to the directory where ftat.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Three levels up from ftat.py
sys.path.append(os.path.join(base_path, "../../"))

from tutorials.algorithms.ses import SES


class ESTEO(SES):
    def __init__(self, mu, lambda_, sigma, bounds, fitness_func, TM_size=200):
        super().__init__(mu, lambda_, sigma, bounds, fitness_func)
        self.features_size = max(1, self.dimension // 3)

        self.TM = []
        self.TM_fitness = []
        self.TM_size = TM_size


    def select_features(self):
        self.feature_to_mutate = np.random.choice(self.features_to_select, self.features_size, replace=False)

    def generate_offsprings(self, base_solution):
        offsprings = []

        for _ in range(self.lambda_):
            self.select_features()
            offspring = self.global_generation(base_solution)
            offsprings.append(offspring)

        return offsprings

    def selection(self, offsprings, fitness):
        for offspring, offspring_fitness in zip(offsprings, fitness):
            if offspring_fitness >= 1: # Discrimination has been found
                self.update_TM(offspring, offspring_fitness)

        best_indices = np.argsort(fitness)[-self.mu:]  # Minimization
        best_offsprings = offsprings[best_indices]
        best_fitness = fitness[best_indices]

        poor_indices = np.where(best_fitness < 1)[0] # All offsprings discriminate

        if len(poor_indices) == 0:
            return best_offsprings

        # Ensure self.TM is a NumPy array
        if len(self.TM) >= len(poor_indices):
            TM_array = np.array(self.TM)  # Convert list to NumPy array

            # Select len(poor_indices) samples from memory
            indices = np.random.choice(len(TM_array), len(poor_indices), replace=False)
            memory_solutions = TM_array[indices]  # Get selected solutions

            # Replace poor-performing solutions
            best_offsprings[poor_indices] = memory_solutions

        elif len(self.TM) > 0: # replace equal number of poor offspring with what is in the memory
            TM_array = np.array(self.TM.copy())
            # Select len(poor_indices) samples from memory
            indices = np.random.choice(len(best_offsprings), len(TM_array), replace=False)
            best_offsprings[indices] = TM_array

        # truncate the thermal memory to the max size
        # self.TM = self.TM[:-self.TM_size]

        return best_offsprings

    def update_TM(self, offspring, offspring_fitness):
        if len(self.TM) >= self.TM_size:
            idx = np.random.randint(self.TM_size)
            if self.TM_fitness[idx] >= offspring_fitness:
                self.TM[idx] = offspring.copy()
                self.TM_fitness[idx] = offspring_fitness
        else:
            self.TM.append(offspring.copy())
            self.TM_fitness.append(offspring_fitness)

    def run(self):
        offsprings = self.get_offsprings()
        fitness = self.evaluate_fitness(offsprings)

        self.population = self.selection(offsprings, fitness)
