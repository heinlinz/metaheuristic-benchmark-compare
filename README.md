# Metaheuristic Optimization: PSO vs GA

A comparison of Particle Swarm Optimization (PSO) and Genetic Algorithm (GA) on standard benchmark functions.

## Algorithms

### PSO - Particle Swarm Optimization
Swarm intelligence algorithm where particles (potential solutions) communicate and move toward the best solution found by the swarm. Features:
- Adaptive inertia weight (decreases over iterations)
- Time-varying acceleration coefficients
- Velocity clamping for stability

### GA - Genetic Algorithm
Evolutionary algorithm inspired by natural selection. Features:
- Tournament selection
- Arithmetic crossover
- Adaptive mutation rate
- Elitism preservation

## Benchmark Functions

| Function | Dimensions | Search Space | Global Minimum |
|----------|------------|---------------|----------------|
| Bohachevsky | N | [-100, 100] | 0 |
| Ackley | N | [-32.768, 32.768] | 0 |
| Rastrigin | N | [-5.12, 5.12] | 0 |
| Power Sum | N | [0, 4] | 0 |

## Installation

```bash
pip install numpy matplotlib scipy scikit-learn
```

## Usage

Run a single optimization:

```bash
python main.py -f ackley -d 10 --pop 30 --iter 50
```

Run with statistical analysis (multiple runs):

```bash
python main.py -f rastrigin -d 10 --pop 30 --iter 50 --runs 30 -o results.txt
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-f/--func` | Benchmark function (boha, ackley, power, rastrigin) | Required |
| `-d/--dim` | Number of dimensions | 2 |
| `--pop` | Population/Swarm size | 30 |
| `--iter` | Number of iterations | 50 |
| `--runs` | Number of runs for statistical analysis | 1 |
| `-o/--output` | Output file for results | None |

## Output

The program generates:
- Convergence curve plot (log scale)
- 3-panel visualization at start, midpoint, and final iterations
- Statistical summary (when `--runs > 1`)

## Project Structure

```
.
├── main.py              # CLI entry point
├── benchmarks.py        # Benchmark function definitions
├── pso_algorithm.py     # PSO implementation
├── ga_algorithm.py     # GA implementation
├── particle.py          # PSO particle class
├── chromosome.py        # GA chromosome class
└── run_experiments.py  # Batch experiment runner
```

## Results

See `*_results_*.txt` files for recorded experiment results on various dimensions.