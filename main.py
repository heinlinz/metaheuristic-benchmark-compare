import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA  # Handles 100D -> 2D visual squashing
from benchmarks import bohachevsky, ackley, power_sum, rastrigin
from pso_algorithm import PSO
from ga_algorithm import GA

def draw_scatter_panels(history, algo_name, map_name, bounds, iterations, dimensions, color):
    """Draws the 3-panel storyboard, using PCA to squash high dimensions to 2D."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{algo_name} on {map_name} Map ({dimensions}D Search → 2D Projection)", fontsize=16)

    turns_to_plot = [0, iterations // 5, iterations - 1]
    titles = ["Start: Exploring...", "Midpoint: Converging...", "Final: Settled"]

    for idx, turn in enumerate(turns_to_plot):
        ax = axes[idx]
        pop_coords = history[turn]['population']
        best_coord = history[turn]['best_parent']

        # --- THE PCA MAGIC ---
        pop_matrix = np.array(pop_coords)
        best_matrix = np.array([best_coord])

        # If the data is more than 2 dimensions, squash it!
        if pop_matrix.shape[1] > 2:
            pca = PCA(n_components=2)
            squashed_pop = pca.fit_transform(pop_matrix)
            squashed_best = pca.transform(best_matrix)
            
            pop_x = squashed_pop[:, 0]
            pop_y = squashed_pop[:, 1]
            best_x, best_y = squashed_best[0][0], squashed_best[0][1]
        else:
            # If it's already 2D, draw it normally
            pop_x = [p[0] for p in pop_coords]
            pop_y = [p[1] for p in pop_coords]
            best_x, best_y = best_coord[0], best_coord[1]
            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[0], bounds[1])
        # ----------------------

        ax.scatter(pop_x, pop_y, c=color, label='Population', zorder=1)
        ax.scatter(best_x, best_y, c='gold', marker='*', s=200, label='Best Position', zorder=2)

        ax.set_title(titles[idx])
        ax.legend()

    plt.tight_layout()

def main():
    parser = argparse.ArgumentParser(description="Run Metaheuristic Optimization Showdown")
    parser.add_argument('-f', '--func', type=str, choices=['boha', 'ackley', 'power', 'rastrigin'], required=True)
    parser.add_argument('--pop', type=int, default=30, help='Population/Swarm size (default: 30)')
    parser.add_argument('--iter', type=int, default=50, help='Number of iterations (default: 50)')
    parser.add_argument('--runs', type=int, default=1, help='Number of independent runs for T-tests (default: 1)')
    parser.add_argument('-d', '--dim', type=int, default=2, help='Number of dimensions (default: 2)')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output file to save results (default: None)')
    args = parser.parse_args()

    # --- Setup ---
    if args.func == 'boha':
        target_function = bohachevsky
        bounds = (-100, 100)
        func_name = "Bohachevsky"
    elif args.func == 'power':
        target_function = power_sum
        bounds = (0, 4)
        func_name = "Power Sum"
    elif args.func == 'rastrigin':
        target_function = rastrigin
        bounds = (-5.12, 5.12)
        func_name = "Rastrigin"
    else: 
        target_function = ackley
        bounds = (-32.768, 32.768)
        func_name = "Ackley"

    print(f"\n--- Initiating Showdown on {func_name} ---")
    print(f"Population: {args.pop} | Iterations: {args.iter} | Dimensions: {args.dim}\n")

    # ==========================================
    # STATISTICAL TRIALS (T-TEST & COST)
    # ==========================================
    if args.runs > 1:
        print(f"Running {args.runs} independent trials in {args.dim}D to calculate statistical significance...")
        pso_scores, ga_scores = [], []
        pso_times, ga_times = [], []

        for r in range(args.runs):
            # Track PSO Cost and Score
            start_pso = time.time()
            pso = PSO(swarm_size=args.pop, bounds=bounds, dimensions=args.dim)
            _, p_score, _, _ = pso.optimize(target_function, iterations=args.iter)
            pso_times.append(time.time() - start_pso)
            pso_scores.append(p_score)

            # Track GA Cost and Score
            start_ga = time.time()
            ga = GA(pop_size=args.pop, bounds=bounds, mutation_rate=0.1, dimensions=args.dim, elitism_count=3)
            _, g_score, _, _ = ga.optimize(target_function, iterations=args.iter)
            ga_times.append(time.time() - start_ga)
            ga_scores.append(g_score)

        # Calculate Averages and T-Test
        avg_pso_time = np.mean(pso_times)
        avg_ga_time = np.mean(ga_times)
        avg_pso_score = np.mean(pso_scores)
        avg_ga_score = np.mean(ga_scores)
        t_stat, p_value = stats.ttest_ind(pso_scores, ga_scores)

        print("\n=== STATISTICAL ANALYSIS RESULTS ===")
        print(f"Cost (Average Runtime): PSO: {avg_pso_time:.5f}s/run | GA: {avg_ga_time:.5f}s/run")
        print(f"Average Accuracy Score: PSO: {avg_pso_score:.8f} | GA: {avg_ga_score:.8f}")
        print(f"T-Test p-value: {p_value:.4e}")
        
        if p_value < 0.05:
            winner = "PSO" if avg_pso_score < avg_ga_score else "GA"
            print(f"Conclusion: The performance difference is statistically significant (p < 0.05). {winner} holds the edge.")
        else:
            print("Conclusion: No statistically significant difference (p >= 0.05).")
        
        if args.output:
            with open(args.output, 'a') as f:
                f.write(f"Function: {func_name} | Dimensions: {args.dim}\n")
                f.write(f"  PSO: avg={avg_pso_score:.8f} ± {np.std(pso_scores):.8f}, time={avg_pso_time:.4f}s, best={min(pso_scores):.8f}\n")
                f.write(f"  GA:  avg={avg_ga_score:.8f} ± {np.std(ga_scores):.8f}, time={avg_ga_time:.4f}s, best={min(ga_scores):.8f}\n")
                f.write(f"  T-test p-value: {p_value:.4e}, Significant: {p_value < 0.05}, Winner: {winner if p_value < 0.05 else 'None'}\n\n")
            print(f"Results saved to {args.output}")
        
        print("\nGenerating visual charts for the final run...")

    # ==========================================
    # ORIGINAL VISUAL RUN
    # ==========================================
    print("Running Particle Swarm Optimization (PSO)...")
    start_pso = time.time()
    pso = PSO(swarm_size=args.pop, bounds=bounds, dimensions=args.dim)
    pso_pos, pso_score, pso_curve, pso_history = pso.optimize(target_function, iterations=args.iter)
    pso_time = time.time() - start_pso

    print("Running Genetic Algorithm (GA)...")
    start_ga = time.time()
    ga = GA(pop_size=args.pop, bounds=bounds, mutation_rate=0.1, dimensions=args.dim)
    ga_pos, ga_score, ga_curve, ga_history = ga.optimize(target_function, iterations=args.iter)
    ga_time = time.time() - start_ga

    # --- Generate Visualizations ---
    draw_scatter_panels(pso_history, "PSO Swarm", func_name, bounds, args.iter, args.dim, color='blue')
    draw_scatter_panels(ga_history, "GA Chromosomes", func_name, bounds, args.iter, args.dim, color='green')

    plt.figure(figsize=(10, 6))
    plt.plot(pso_curve, color='blue', linewidth=2, label='PSO')
    plt.plot(ga_curve, color='green', linewidth=2, label='GA')
    
    plt.yscale('log') 
    plt.title(f'Performance Comparison: PSO vs GA on {func_name} ({args.dim}D, Log Scale)', fontsize=14)
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Best Fitness Score (Log Scale)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    if args.output and args.runs <= 1:
        winner = 'PSO' if pso_score < ga_score else 'GA'
        with open(args.output, 'a') as f:
            f.write(f"=== STATISTICAL ANALYSIS RESULTS ===\n")
            f.write(f"Cost (Average Runtime): PSO: {pso_time:.5f}s/run | GA: {ga_time:.5f}s/run\n")
            f.write(f"Average Accuracy Score: PSO: {pso_score:.8f} | GA: {ga_score:.8f}\n")
            f.write(f"T-Test p-value: N/A (requires --runs > 1)\n")
            f.write(f"Conclusion: Single run only. Winner: {winner}\n\n")
        print(f"Results saved to {args.output}")
    
    print("\nDisplaying graphs. Close the windows to exit the program.")
    plt.show()

if __name__ == "__main__":
    main()
