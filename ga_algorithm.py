import numpy as np
from chromosome import Chromosome

class GA:
    def __init__(self, pop_size, bounds, mutation_rate=0.1, dimensions=2, elitism_count=3):
        self.pop_size = pop_size
        self.bounds = bounds
        self.mutation_rate = mutation_rate
        self.dimensions = dimensions
        self.elitism_count = elitism_count
        
        self.population = [Chromosome(bounds, dimensions=dimensions) for _ in range(pop_size)]
        
        self.global_best_fitness = float('inf')
        self.global_best_genes = None

    def _tournament_selection(self, tournament_size=3):
        candidates = []
        for _ in range(tournament_size):
            idx = np.random.randint(0, self.pop_size)
            candidates.append(self.population[idx])
        return min(candidates, key=lambda x: x.fitness)

    def _crossover(self, parent1, parent2):
        alpha = np.random.uniform(0, 1)
        child_genes = alpha * parent1.genes + (1 - alpha) * parent2.genes
        return child_genes

    def _mutate(self, child_genes, best_fitness):
        range_size = self.bounds[1] - self.bounds[0]
        adaptive_mut = self.mutation_rate * (1 + best_fitness / (best_fitness + 1))
        
        if np.random.rand() < adaptive_mut:
            mutation = np.random.uniform(-1, 1, self.dimensions) * range_size * 0.05
            child_genes = child_genes + mutation
            child_genes = np.clip(child_genes, self.bounds[0], self.bounds[1])
        
        return child_genes

    def optimize(self, objective_function, iterations=50):
        convergence_curve = []
        history = {}
        snapshots = [0, iterations // 5, iterations - 1]

        for i in range(iterations):
            for chromo in self.population:
                chromo.evaluate(objective_function)
                
                if chromo.fitness < self.global_best_fitness:
                    self.global_best_fitness = chromo.fitness
                    self.global_best_genes = np.copy(chromo.genes)

            self.population.sort(key=lambda x: x.fitness)

            new_population = []
            for j in range(self.elitism_count):
                new_population.append(Chromosome(self.bounds, np.copy(self.population[j].genes), self.dimensions))

            while len(new_population) < self.pop_size:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()

                child_genes = self._crossover(parent1, parent2)
                child_genes = self._mutate(child_genes, self.global_best_fitness)

                new_population.append(Chromosome(self.bounds, child_genes, self.dimensions))

            self.population = new_population
            convergence_curve.append(self.global_best_fitness)

            if i in snapshots:
                pop_coords = [np.copy(chromo.genes) for chromo in self.population]
                history[i] = {
                    'population': pop_coords,
                    'best_parent': np.copy(self.global_best_genes)
                }

        return self.global_best_genes, self.global_best_fitness, convergence_curve, history