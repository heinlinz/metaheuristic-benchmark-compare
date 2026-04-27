import numpy as np
from particle import Particle
class PSO:
    # NEW: The manager accepts the dimensions and passes them to the particles
    def __init__(self, swarm_size, bounds, dimensions=2):
        self.bounds = bounds
        self.dimensions = dimensions
        self.swarm = [Particle(bounds, dimensions) for _ in range(swarm_size)]
        
        self.global_best_score = float('inf')
        self.global_best_position = None

    def optimize(self, objective_function, iterations=100, w=0.5, c1=1.5, c2=1.5):
        convergence_curve = []
        history = {}
        snapshots = [0, iterations // 5, iterations - 1]

        for i in range(iterations):
            w_adapted = 0.9 - (0.9 - 0.4) * i / iterations
            c1_val = 2.0 - 1.5 * i / iterations
            c2_val = 1.0 + 1.5 * i / iterations
            
            for particle in self.swarm:
                particle.evaluate(objective_function)
                
                if particle.best_score < self.global_best_score:
                    self.global_best_position = np.copy(particle.best_position)
                    self.global_best_score = particle.best_score
            
            for particle in self.swarm:
                particle.update_velocity(self.global_best_position, w_adapted, c1_val, c2_val, i, iterations)
                particle.update_position(self.bounds)
            
            convergence_curve.append(self.global_best_score)
            
            if i in snapshots:
                pop_coords = [np.copy(particle.position) for particle in self.swarm]
                history[i] = {
                    'population': pop_coords,
                    'best_parent': np.copy(self.global_best_position)
                }

        return self.global_best_position, self.global_best_score, convergence_curve, history
