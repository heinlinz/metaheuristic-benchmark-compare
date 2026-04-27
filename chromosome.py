import numpy as np

class Chromosome:
    def __init__(self, bounds, genes=None, dimensions=2):
        if genes is None:
            # NEW: The chromosome generates N-dimensions of random genes
            self.genes = np.random.uniform(bounds[0], bounds[1], dimensions)
        else:
            self.genes = np.copy(genes)
            
        self.fitness = float('inf')
        self.bounds = bounds

    def evaluate(self, objective_function):
        self.fitness = objective_function(self.genes)
