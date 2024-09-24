# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    pdfs.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/02/29 14:56:47 by daniloceano       #+#    #+#              #
#    Updated: 2024/09/24 10:49:12 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joypy import joyplot
from tqdm import tqdm
from scipy.stats import gaussian_kde

PATH = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic'
base_path = f'{PATH}/csv_database_energy_by_periods'
output_directory = f'./figures/pdfs/'

COLOR_PHASES = {
    'Total': '#070A2B',
    'incipient': '#65a1e6',
    'intensification': '#f7b538',
    'intensification 2': '#ca6702',
    'mature': '#d62828',
    'mature 2': '#9b2226',
    'decay': '#9aa981',
    'decay 2': '#386641',
}

COLOR_TERMS = ["#3B95BF", "#87BF4B", "#BFAB37", "#BF3D3B", "#873e23", "#A13BF0"]

quantile_caps = {
    # 'Kz': (0, 0.8),
    'Cz': (0.2, 0.9),
    'Ck': (0.2, 0.9),
    # 'Ca': (0.2, 0.9),
    'Ce': (0.01, 0.99),
    'BAz': (0.2, 0.9),
    'BAe': (0.2, 0.9),
    'BKz': (0.2, 0.9),
    'BKe': (0.2, 0.9),
    'RKz': (0.3, 0.7),
    'RKe': (0.3, 0.95),
    '∂Ke/∂t (finite diff.)': (0.3, 0.7),
    '∂Kz/∂t (finite diff.)': (0.3, 0.7),
    '∂Az/∂t (finite diff.)': (0.3, 0.7),
    '∂Ae/∂t (finite diff.)': (0.3, 0.7),
}

LEGEND_POSITION = {
    'Energy Terms': 'upper right',
    'Conversion Terms': 'upper right',
    'Boundary Terms': 'upper right',
    'Pressure Work Terms': 'upper right',
    'Generation/Residual Terms': 'best',
    'Budget Terms': 'best'
}

# Define x-axis limits for each group
AXIS_LIMITS = {
    'Energy Terms': (0, 1e6),
    'Conversion Terms': (-10, 10),
    'Boundary Terms': (-15, 15),
    'Pressure Work Terms': (-200, 250),
    'Generation/Residual Terms': (-20, 10),
    'Budget Terms': (-5, 5)
}

LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

# Global variables for font sizes
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 12
TITLE_FONT_SIZE = 16
LEGEND_FONT_SIZE = 12

def read_life_cycles(base_path):
    """
    Reads all CSV files in the specified directory and collects DataFrame for each system.
    """
    systems_energetics = {}

    for filename in tqdm(os.listdir(base_path), desc="Reading CSV files"):
        if filename.endswith('.csv'):
            file_path = os.path.join(base_path, filename)
            system_id = filename.split('_')[0]
            try:
                df = pd.read_csv(file_path)
                systems_energetics[system_id] = df
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return systems_energetics

def compute_group_caps(systems_energetics, terms_prefix, special_case=None):
    """
    Computes caps for a group of terms based on the 0.2 and 0.8 quantiles across all systems.

    Parameters:
    - systems_energetics: Dictionary of DataFrames with system data.
    - terms_prefix: Prefixes of terms to include in the plot.
    - special_case: Special handling for certain groups (e.g., 'Energy Terms').

    Returns:
    - A tuple (min_cap, max_cap) representing the computed value caps for the group.
    """
    all_values = []

    for df in systems_energetics.values():
        relevant_cols = [col for col in df.columns if col.startswith(tuple(terms_prefix))]
        all_values.extend(df[relevant_cols].values.flatten())

    all_values = pd.Series(all_values).dropna()

    if special_case == 'Energy Terms':
        min_cap = 0  # Special case for Energy Terms
        max_cap = all_values.quantile(0.8)
    else:
        q2 = all_values.quantile(0.2)
        q8 = all_values.quantile(0.8)
        highest_cap = np.amax(np.abs([q2, q8]))
        min_cap, max_cap = -highest_cap, highest_cap

    return min_cap, max_cap

def plot_group_panel(systems_energetics, groups, output_directory):
    """
    Plots ridge plots for all term groups in a single figure with multiple panels.
    
    Parameters:
    - systems_energetics: Dictionary of DataFrames with system data.
    - groups: Dictionary of groups with their corresponding prefixes.
    - output_directory: Directory to save the final combined plot.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # Adjusted figure size for better layout

    # Get the terms for the group
    terms = systems_energetics[list(systems_energetics.keys())[0]].drop(columns=['Unnamed: 0']).columns.to_list()

    # Group the terms using the prefixes
    term_groups = {}
    for term in terms:
        for group_name, terms_prefix in groups.items():
            if term.startswith(tuple(terms_prefix)):
                if group_name not in term_groups:
                    term_groups[group_name] = []
                term_groups[group_name].append(term)
    
    for idx, (group_name, terms_prefix) in enumerate(groups.items()):
        ax = axes[idx // 3, idx % 3]
        
        # Concatenate all systems' dataframes while retaining the system id and phase
        all_data = pd.concat([df.assign(system_id=system_id) for system_id, df in systems_energetics.items()])
        all_data.rename(columns={'Unnamed: 0': 'Phase'}, inplace=True)
        
        # Convert relevant columns to numeric, forcing errors to NaN
        relevant_columns = ['system_id'] + [col for col in all_data.columns if col.startswith(tuple(terms_prefix))]
        all_data[relevant_columns] = all_data[relevant_columns].apply(pd.to_numeric, errors='coerce')
        
        # Compute mean across all phases for each system
        mean_data = all_data.drop('Phase', axis=1).groupby('system_id').mean().reset_index()
        
        # Melt the dataframe for plotting
        mean_data_melted = mean_data.melt(id_vars=['system_id'], var_name='Term', value_name='Value')
        
        # Plot the terms for the group
        for idy, term in enumerate(term_groups[group_name]):
            term_data = mean_data_melted[mean_data_melted['Term'] == term]

            if 'Kz' in term:
                term_data['Value'] = term_data['Value'] / 10

            if not term_data.empty:
                # Set the label based on the term name
                label = term if '∂' not in term else term.split('(finite diff.)')[0]
                label = label+'*' if 'Kz' in label else label
                sns.kdeplot(data=term_data, x='Value', label=label,
                            ax=ax, bw_adjust=0.5, fill=False,
                            color=COLOR_TERMS[idy], linewidth=2, alpha=0.8)

        # Set axis limits based on the group
        ax.set_xlim(AXIS_LIMITS[group_name])

        # Add vertical line at 0
        if group_name != 'Energy Terms':
            ax.axvline(0, color='gray', linestyle='--', linewidth=1, zorder=0)

        legend_position = LEGEND_POSITION.get(group_name, 'best')  # Use 'best' if no position is defined
        ax.legend(fontsize=LEGEND_FONT_SIZE, loc=legend_position)

        ax.set_title(f'({LABELS[idx]}) {group_name}', fontsize=TITLE_FONT_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
        ax.set_xlabel('', fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel('', fontsize=LABEL_FONT_SIZE)

    plt.tight_layout()
    combined_plot_filename = 'combined_ridge.png'
    combined_plot_path = os.path.join(output_directory, combined_plot_filename)
    plt.savefig(combined_plot_path, bbox_inches='tight')  # Ensure everything, including legends, fits in the figure
    plt.close()
    print(f"Saved combined plot as {combined_plot_filename} in {output_directory}")

if __name__ == "__main__":
    os.makedirs(output_directory, exist_ok=True)

    systems_energetics = read_life_cycles(base_path)

    # Define term prefixes for each group
    groups = {
        'Energy Terms': ['A', 'K'],
        'Conversion Terms': ['C'],
        'Boundary Terms': ['BA', 'BK'],
        'Pressure Work Terms': ['BΦ'],
        'Generation/Residual Terms': ['G', 'R'],
        'Budget Terms': ['∂']
    }

    plot_group_panel(systems_energetics, groups, output_directory)