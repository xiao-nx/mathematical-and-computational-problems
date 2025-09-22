import os
import math
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib.pyplot as plt
from scipy.special import gamma
import pprint
import argparse

# --- Step 1: Data Generation Function ---
def generate_random_points(n_points):
    """
    Generates n random points (x, y) with independent standard normal coordinates.

    Parameters:
    n_points (int): The number of points to generate.

    Returns:
    tuple: A tuple containing two numpy arrays for x and y coordinates.
    """
    # Generate x and y coordinates independently from a standard normal distribution N(0, 1)
    x = np.random.randn(n_points)
    y = np.random.randn(n_points)
    return x, y

# --- Step 2: Main Simulation Function ---
def simulate_pearson_correlation(n_points, num_repeats):
    """
    Simulates the distribution of the Pearson correlation coefficient by
    repeating the process of generating n random points and calculating r.

    Parameters:
    n_points (int): The number of points (sample size n) for each calculation.
    num_repeats (int): The number of times to repeat the simulation (r).

    Returns:
    list: A list containing `num_repeats` correlation coefficient values.
    """
    correlation_coefficients = []

    for _ in range(num_repeats):
        # Use the dedicated function to generate data points
        x, y = generate_random_points(n_points)

        # Calculate the Pearson correlation coefficient
        # np.corrcoef returns a 2x2 matrix; we need the off-diagonal element
        corr_matrix = np.corrcoef(x, y)
        r_val = corr_matrix[0, 1]
        
        correlation_coefficients.append(r_val)

    return correlation_coefficients

# --- Theoretical Probability Density Function (for validation) ---
def r_pdf_rho_0(r, n):
    """
    Calculates the theoretical probability density function (PDF) for r when
    the population correlation rho is 0.

    Parameters:
    r (float or array): The correlation coefficient value(s).
    n (int): The sample size.

    Returns:
    float or array: The corresponding probability density.
    """
    if n <= 2:
        return np.nan

    numerator = gamma((n - 1) / 2)
    denominator = math.sqrt(math.pi) * gamma((n - 2) / 2)

    r = np.clip(r, -1, 1)
    return (numerator / denominator) * (1 - r**2)**((n - 4) / 2)


# --- Analyze and Save Results ---
def analyze_results(n_values, num_repeats):
    """
    Runs the simulation, calculates key statistics, and saves the results to a CSV file.

    Parameters:
    n_values (list): A list of sample sizes (n) to analyze.
    num_repeats (int): The number of times to repeat the simulation.
    """
    simulation_results = []
    
    print("\n--- Analyzing Simulation Results ---")

    for n in n_values:
        if n < 3:
            print(f"Skipping n = {n} as it's not applicable for this analysis.")
            continue
            
        correlations = simulate_pearson_correlation(n, num_repeats)
        
        # Calculate statistics from simulation
        mean_r = np.mean(correlations)
        var_r = np.var(correlations)
        
        # Apply Fisher z-transformation
        # Note: atanh(x) = 0.5 * log((1+x)/(1-x))
        # Clip to avoid infinite values for r=1 or r=-1
        clipped_correlations = np.clip(correlations, -0.999999, 0.999999)
        z_transformed = np.arctanh(clipped_correlations)
        mean_z = np.mean(z_transformed)
        var_z = np.var(z_transformed)
        
        # Theoretical variance of Fisher z-transform
        theory_var_z = 1 / (n - 3)
        
        simulation_results.append({
            'n': n,
            'mean(r)': mean_r,
            'var(r)': var_r,
            'mean(z)': mean_z,
            'var(z)': var_z,
            'theory_var_z': theory_var_z
        })
        
        print(f"n = {n}: Mean(r) = {mean_r:.4f}, Var(r) = {var_r:.4f}, Var(z) = {var_z:.4f}, Theory Var(z) = {theory_var_z:.4f}")

    return simulation_results
    
    
def bootstrap_analysis(data, num_bootstraps=1000):
    """
    Performs bootstrap resampling to estimate confidence intervals for the mean.

    Parameters:
    data (list or array): The list of correlation coefficients from the simulation.
    num_bootstraps (int): The number of bootstrap samples to draw.

    Returns:
    tuple: A tuple containing the mean of bootstrap means and the 95% confidence interval.
    """

    bootstrap_means = []
    n_data = len(data)
    
    for _ in range(num_bootstraps):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=n_data, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    # Calculate confidence interval
    lower_bound = np.percentile(bootstrap_means, 2.5)
    upper_bound = np.percentile(bootstrap_means, 97.5)
          
    return np.mean(bootstrap_means), (lower_bound, upper_bound)


# --- Main function to run the entire simulation process ---
def main(args):
    """
    Main function to run the simulation, plotting, and analysis based on parsed arguments.
    """
    
    if os.path.exists(args.save_results_path) is False:
        os.makedirs(args.save_results_path)
    
    # Section 1: Visualization of the distributions
    if not args.skip_plot:
        # Section 1: Visualization of the distributions - Refactored for adaptability
        
        num_plots = len(args.n_values)
        
        # Determine the number of rows and columns for a square-like layout
        # Using a fixed number of columns (e.g., 3 or 4) is also a good strategy
        # num_cols = int(np.ceil(np.sqrt(num_plots)))
        # num_rows = int(np.ceil(num_plots / num_cols))
        num_rows = 1
        num_cols = 3
        
        # Adjust figure size dynamically
        # fig_width = num_cols * 5
        # fig_height = num_rows * 10
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, 
                                 figsize=(12, 5), sharey=True)

        # Flatten the axes array for easy iteration, especially for single-row/column cases
        axes = axes.flatten()

        for i, n in enumerate(args.n_values):
            print(f"Running simulation for n = {n}...")
            correlations = simulate_pearson_correlation(n, args.num_repeats)
            ax = axes[i]
            
            # Plot simulation and theoretical PDF
            ax.hist(correlations, bins=50, density=True, alpha=0.6, label='Simulation')
            r_range = np.linspace(-1, 1, 400)
            pdf_vals = [r_pdf_rho_0(r, n) for r in r_range]
            ax.plot(r_range, pdf_vals, 'r-', lw=2, label='Theoretical PDF')
            
            # Set title and labels
            ax.set_title(f'n = {n}', fontsize=14)
            ax.set_xlabel('Correlation Coefficient (r)')
        
        
        # Hide any unused subplots if the total number is not a perfect square
        for i in range(num_plots, len(axes)):
            fig.delaxes(axes[i])
            
        axes[0].set_ylabel('Density')
        # axes[-1].legend()
        axes[-1].legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=12)
        fig.suptitle(f'Distribution of Pearson\'s Correlation Coefficient for {args.num_repeats:,} Repeats', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(args.save_results_path, 'correlation_distribution10-50-100.png'))
        # plt.show()


    # # Section 2: Quantitative Analysis and CSV Output
    # if not args.skip_csv:
    #     simulation_results = analyze_results(args.n_values, args.num_repeats)
        
    # # Section 3: Bootstrap Uncertainty Analysis
    # boostrap_results= []
    # for n in args.n_values:
    #     if n < 3:
    #         continue
        
    #     correlations = simulate_pearson_correlation(n, args.num_repeats)
        
    #     # Analyze the uncertainty of the mean(r)
    #     bootstrap_means, bootstrap_cr = bootstrap_analysis(correlations)
    #     boostrap_results.append({
    #         'n': n, 
    #         'Bootstrap Mean': round(bootstrap_means, 4),
    #         '95% Confidence Interval for Mean(r)': (round(bootstrap_cr[0], 4), round(bootstrap_cr[1], 4))
    #     })
    # pprint.pprint(boostrap_results, indent=4, width=100, sort_dicts=False)

    # # Convert list to DataFrame and save to CSV
    # simulation_df = pd.DataFrame(simulation_results)
    # boostrap_df = pd.DataFrame(boostrap_results)
    # results_df = pd.merge(simulation_df, boostrap_df, on='n', how='inner')    
    # results_df.to_csv(os.path.join(args.save_results_path, 'results.csv'), index=False)
    # print("\nResults have been saved to 'simulation_results.csv'")


# --- Main Program to Run Simulation and Plot Results ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Pearson's correlation distribution and analyze results.")
    parser.add_argument('--n_values', type=int, nargs='+', default=[10, 50, 100],
                        help='A list of sample sizes (n) to simulate.[4, 5, 6]')    
    parser.add_argument('--num_repeats', type=int, default=100000,
                        help='The number of simulation repeats for each n.')
    parser.add_argument('--skip_plot', action='store_true',
                        help='If specified, skips the plotting of histograms.')
    parser.add_argument('--skip_csv', action='store_true',
                        help='If specified, skips the analysis and saving to CSV.')    
    parser.add_argument('--save_results_path', default='./results/',
                        help='If specified, skips the analysis and saving to CSV.')        
    
    args = parser.parse_args()
    
    args = parser.parse_args()
    
    main(args)