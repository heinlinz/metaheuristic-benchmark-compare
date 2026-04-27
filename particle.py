import numpy as np


class Particle:
    def __init__(self, bounds, dimensions):
        # NEW: The particle spawns in N-dimensions based on the command line
        self.position = np.random.uniform(bounds[0], bounds[1], dimensions)
        self.velocity = np.random.uniform(-1, 1, dimensions)
        
        self.best_position = np.copy(self.position)
        self.best_score = float('inf')

    def evaluate(self, objective_function):
        score = objective_function(self.position)
        if score < self.best_score:
            self.best_position = np.copy(self.position)
            self.best_score = score

    def update_velocity(self, global_best_position, w, c1, c2, iteration=None, total_iterations=None):
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        
        if iteration is not None and total_iterations is not None:
            progress = iteration / total_iterations
            c1_adapt = c1 * (1 - progress * 0.5)
            c2_adapt = c2 * (0.5 + progress * 0.5)
        else:
            c1_adapt = c1
            c2_adapt = c2
        
        cognitive = c1_adapt * r1 * (self.best_position - self.position)
        social = c2_adapt * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, bounds):
        range_size = bounds[1] - bounds[0]
        v_max = 0.2 * range_size
        self.velocity = np.clip(self.velocity, -v_max, v_max)
        self.position += self.velocity
        self.position = np.clip(self.position, bounds[0], bounds[1])
