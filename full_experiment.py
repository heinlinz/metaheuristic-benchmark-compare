import argparse
import time
import numpy as np
from scipy import stats
from benchmarks import bohachevsky, ackley, power_sum, rastrigin
from pso_algorithm import PSO
from ga_algorithm import GA

def run_single_test(func_name, target_function, bounds, runs=30, pop=30, iterations=500, dimensions=10):
    np.random.seed(42)
    
    pso_scores, ga_scores = [], []
    pso_times, ga_times = [], []
    pso_curves, ga_curves = [], []
    
    for r in range(runs):
        # PSO
        start_pso = time.time()
        pso = PSO(swarm_size=pop, bounds=bounds, dimensions=dimensions)
        _, p_score, p_curve, _ = pso.optimize(target_function, iterations=iterations)
        pso_times.append(time.time() - start_pso)
        pso_scores.append(p_score)
        pso_curves.append(p_curve)
        
        # GA
        start_ga = time.time()
        ga = GA(pop_size=pop, bounds=bounds, mutation_rate=0.1, dimensions=dimensions)
        _, g_score, g_curve, _ = ga.optimize(target_function, iterations=iterations)
        ga_times.append(time.time() - start_ga)
        ga_scores.append(g_score)
        ga_curves.append(g_curve)
    
    avg_pso = np.mean(pso_scores)
    std_pso = np.std(pso_scores)
    avg_ga = np.mean(ga_scores)
    std_ga = np.std(ga_scores)
    
    avg_pso_time = np.mean(pso_times)
    avg_ga_time = np.mean(ga_times)
    
    # Find convergence iteration (90% of best)
    def find_convergence(curve, best_val):
        if best_val <= 0:
            return iterations
        threshold = best_val * 1.1
        for i, v in enumerate(curve):
            if v <= threshold:
                return i
        return iterations
    
    # Compute average convergence iteration (approximate)
    pso_final = np.min(pso_scores)
    ga_final = np.min(ga_scores)
    
    t_stat, p_value = stats.ttest_ind(pso_scores, ga_scores)
    
    return {
        'func': func_name, 'dim': dimensions,
        'pso': {'avg': avg_pso, 'std': std_pso, 'time': avg_pso_time, 'final': pso_final},
        'ga': {'avg': avg_ga, 'std': std_ga, 'time': avg_ga_time, 'final': ga_final},
        'p_value': p_value, 'significant': p_value < 0.05,
        'better': 'PSO' if avg_pso < avg_ga else 'GA',
        'pso_curve': pso_curves, 'ga_curve': ga_curves
    }

def main():
    parser = argparse.ArgumentParser(description="Full Experiment Suite")
    parser.add_argument('--pop', type=int, default=30)
    parser.add_argument('--iter', type=int, default=500)
    parser.add_argument('--runs', type=int, default=30)
    args = parser.parse_args()
    
    benchmarks = [
        ('Ackley', ackley, (-32.768, 32.768)),
        ('Rastrigin', rastrigin, (-5.12, 5.12)),
        ('Bohachevsky', bohachevsky, (-100, 100)),
    ]
    
    dims = [2, 5, 10, 20]
    
    results = []
    for dim in dims:
        for name, func, bounds in benchmarks:
            print(f"Testing {name} at {dim}D...", flush=True)
            r = run_single_test(name, func, bounds, args.runs, args.pop, args.iter, dim)
            results.append(r)
    
    # Print results table
    print("\n" + "="*90)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*90)
    print(f"{'Function':<15} {'Dim':>4} | {'PSO Avg':>15} | {'GA Avg':>15} | {'p-value':>12} | {'Winner':>6}")
    print("-"*90)
    
    for r in results:
        print(f"{r['func']:<15} {r['dim']:>4} | {r['pso']['avg']:>15.4f} | {r['ga']['avg']:>15.4f} | {r['p_value']:>12.2e} | {r['better']:>6}")
    
    # Save detailed results
    with open('full_results.txt', 'w') as f:
        f.write("METAHEURISTIC OPTIMIZATION - FULL REPORT\n")
        f.write(f"Config: pop={args.pop}, iter={args.iter}, runs={args.runs}\n\n")
        f.write("="*90 + "\n")
        f.write("DETAILED RESULTS BY FUNCTION AND DIMENSION\n")
        f.write("="*90 + "\n\n")
        
        for r in results:
            f.write(f"Function: {r['func']} | Dimensions: {r['dim']}\n")
            f.write(f"  PSO: avg={r['pso']['avg']:.6f} ± {r['pso']['std']:.6f}, time={r['pso']['time']:.4f}s, best={r['pso']['final']:.6f}\n")
            f.write(f"  GA:  avg={r['ga']['avg']:.6f} ± {r['ga']['std']:.6f}, time={r['ga']['time']:.4f}s, best={r['ga']['final']:.6f}\n")
            f.write(f"  T-test p-value: {r['p_value']:.4e}, Significant: {r['significant']}, Winner: {r['better']}\n\n")
        
        f.write("\n" + "="*90 + "\n")
        f.write("CONFIGURATION\n")
        f.write("="*90 + "\n")
        f.write(f"Population size: {args.pop}\n")
        f.write(f"Iterations: {args.iter}\n")
        f.write(f"Number of runs: {args.runs}\n")
        f.write(f"Dimensions tested: {dims}\n")
    
    print(f"\nResults saved to full_results.txt")

if __name__ == "__main__":
    main()