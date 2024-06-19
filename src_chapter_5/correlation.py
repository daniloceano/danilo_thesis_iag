import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import gaussian_kde
from matplotlib.colors import BoundaryNorm

PATH = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic'
base_path = f'{PATH}/csv_database_energy_by_periods'
tracks_dir = f'{PATH}/tracks_SAt'
output_directory = f'../figures_chapter_5/correlation/'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def read_life_cycles(base_path):
    """
    Reads all CSV files in the specified directory and collects DataFrame for each system.
    """
    systems_energetics = {}

    for filename in tqdm(os.listdir(base_path)[:100], desc="Reading CSV files"):
        if filename.endswith('.csv'):
            file_path = os.path.join(base_path, filename)
            system_id = filename.split('_')[0]
            try:
                df = pd.read_csv(file_path)
                df = df.rename(columns=lambda x: x.replace(' (finite diff.)', ''))
                systems_energetics[system_id] = df
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return systems_energetics

def compute_mean_values(systems_energetics):
    mean_values = {}
    for system_id, df in systems_energetics.items():
        numeric_df = df.select_dtypes(include=[np.number])
        mean_values[system_id] = numeric_df.mean()
    return pd.DataFrame(mean_values).T

# Function to plot and save the correlation matrix
def plot_correlation_matrix(df, title, filename):
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    # Create discrete color scale with intervals of 0.2
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    boundaries = np.arange(-1, 1.2, 0.2)
    norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)
    ax = sns.heatmap(correlation_matrix, cmap=cmap, norm=norm, cbar_kws={'ticks': boundaries})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_correlation_matrices_per_phase(systems_energetics):
    all_phases = ['incipient', 'intensification', 'mature', 'decay']
    phase_data = {phase: [] for phase in all_phases}

    for df in systems_energetics.values():
        for phase in all_phases:
            phase_df = df[df['Unnamed: 0'] == phase]
            if not phase_df.empty:
                numeric_phase_df = phase_df.select_dtypes(include=[np.number])
                phase_data[phase].append(numeric_phase_df.mean())

    for phase, data in phase_data.items():
        if data:
            phase_df = pd.DataFrame(data)
            filename = os.path.join(output_directory, f'correlation_matrix_{phase}.png')
            plot_correlation_matrix(phase_df, f'Correlation Matrix for {phase.capitalize()} Phase', filename)

systems_energetics = read_life_cycles(base_path)

# Compute mean values for each system
mean_values_df = compute_mean_values(systems_energetics)

# Plot and save correlation matrix using the mean values across all phases
plot_correlation_matrix(mean_values_df, 'Correlation Matrix Using Mean Values Across All Phases', os.path.join(output_directory, 'correlation_matrix_mean_values.png'))

# Plot and save correlation matrices for each phase
plot_correlation_matrices_per_phase(systems_energetics)