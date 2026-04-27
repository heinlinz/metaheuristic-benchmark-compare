import argparse
import time
import numpy as np
from scipy import stats
from benchmarks import bohachevsky, ackley, power_sum, rastrigin
from pso_algorithm import PSO
from ga_algorithm import GA

def run_experiment(func_name, target_function, bounds, runs=30, pop=30, iterations=200, dimensions=10):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {func_name} | {dimensions}D | {runs} runs")
    print(f"{'='*60}")
    
    pso_scores, ga_scores = [], []
    pso_times, ga_times = [], []
    pso_convergence, ga_convergence = [], []
    
    for r in range(runs):
        if r % 10 == 0:
            print(f"  Run {r+1}/{runs}...", end=" ", flush=True)
        
        # PSO
        start_pso = time.time()
        pso = PSO(swarm_size=pop, bounds=bounds, dimensions=dimensions)
        _, p_score, p_curve, _ = pso.optimize(target_function, iterations=iterations)
        pso_times.append(time.time() - start_pso)
        pso_scores.append(p_score)
        pso_convergence.append(len(p_curve))
        
        # GA
        start_ga = time.time()
        ga = GA(pop_size=pop, bounds=bounds, mutation_rate=0.1, dimensions=dimensions)
        _, g_score, g_curve, _ = ga.optimize(target_function, iterations=iterations)
        ga_times.append(time.time() - start_ga)
        ga_scores.append(g_score)
        ga_convergence.append(len(g_curve))
    
    print(f"\n  Completed!")
    
    # Statistics
    avg_pso = np.mean(pso_scores)
    std_pso = np.std(pso_scores)
    avg_ga = np.mean(ga_scores)
    std_ga = np.std(ga_scores)
    
    avg_pso_time = np.mean(pso_times)
    avg_ga_time = np.mean(ga_times)
    
    # T-test
    t_stat, p_value = stats.ttest_ind(pso_scores, ga_scores)
    
    # When did each converge (first iteration within 1% of best)?
    def find_convergence_iteration(curve, final_score, threshold=0.01):
        if final_score == 0 or final_score is None:
            return iterations
        target = final_score * (1 + threshold)
        for i, v in enumerate(curve):
            if v <= target:
                return i
        return iterations
    
    pso_conv_iters = [find_convergence_iteration(p_curve, np.min(pso_scores)) for p_curve in [pso.optimize(target_function, iterations=iterations)[2] for _ in range(1)]][0]
    ga_conv_iters = [find_convergence_iteration(g_curve, np.min(ga_scores)) for g_curve in [ga.optimize(target_function, iterations=iterations)[2] for _ in range(1)]][0]
    
    results = {
        'func': func_name,
        'dim': dimensions,
        'pso': {'avg': avg_pso, 'std': std_pso, 'time': avg_pso_time},
        'ga': {'avg': avg_ga, 'std': std_ga, 'time': avg_ga_time},
        'p_value': p_value,
        'significant': p_value < 0.05,
        'better': 'PSO' if avg_pso < avg_ga else 'GA'
    }
    
    print(f"\n  RESULTS:")
    print(f"  PSO: avg={avg_pso:.6f} ± {std_pso:.6f}, time={avg_pso_time:.4f}s")
    print(f"  GA:  avg={avg_ga:.6f} ± {std_ga:.6f}, time={avg_ga_time:.4f}s")
    print(f"  T-test p-value: {p_value:.4e}, significant: {p_value < 0.05}")
    print(f"  Winner: {'PSO' if avg_pso < avg_ga else 'GA'}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Metaheuristic Optimization Experiments")
    parser.add_argument('--runs', type=int, default=30, help='Number of runs per test')
    parser.add_argument('--pop', type=int, default=30, help='Population size')
    parser.add_argument('--iter', type=int, default=200, help='Iterations')
    parser.add_argument('-d', '--dim', type=int, default=10, help='Dimensions')
    args = parser.parse_args()
    
    benchmarks = [
        ('Ackley', ackley, (-32.768, 32.768)),
        ('Rastrigin', rastrigin, (-5.12, 5.12)),
        ('Bohachevsky', bohachevsky, (-100, 100)),
    ]
    
    all_results = []
    for name, func, bounds in benchmarks:
        result = run_experiment(name, func, bounds, 
                               runs=args.runs, pop=args.pop, 
                               iterations=args.iter, dimensions=args.dim)
        all_results.append(result)
    
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Function':<15} {'Dim':<5} {'PSO Avg':<15} {'GA Avg':<15} {'p-value':<12} {'Winner':<6}")
    print("-"*60)
    for r in all_results:
        print(f"{r['func']:<15} {r['dim']:<5} {r['pso']['avg']:<15.6f} {r['ga']['avg']:<15.6f} {r['p_value']:<12.4e} {r['better']:<6}")
    
    print(f"\nConfig: pop={args.pop}, iter={args.iter}, runs={args.runs}")
    print(f"Results saved to experiments.txt")
    
    # Save to file
    with open('experiments.txt', 'w') as f:
        f.write("METAHEURISTIC OPTIMIZATION EXPERIMENTAL RESULTS\n")
        f.write(f"Config: pop={args.pop}, iter={args.iter}, runs={args.runs}, dim={args.dim}\n\n")
        f.write(f"{'Function':<15} {'Dim':<5} {'PSO Avg':<15} {'GA Avg':<15} {'p-value':<12} {'Winner':<6}\n")
        f.write("-"*60 + "\n")
        for r in all_results:
            f.write(f"{r['func']:<15} {r['dim']:<5} {r['pso']['avg']:<15.6f} {r['ga']['avg']:<15.6f} {r['p_value']:<12.4e} {r['better']:<6}\n")

if __name__ == "__main__":
    main()