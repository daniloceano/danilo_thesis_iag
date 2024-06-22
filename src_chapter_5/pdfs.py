# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    pdfs.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/02/29 14:56:47 by daniloceano       #+#    #+#              #
#    Updated: 2024/06/22 20:26:27 by daniloceano      ###   ########.fr        #
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

PATH = '/Users/danilocoutodesouza/Documents/Programs_and_scripts/energetic_patterns_cyclones_south_atlantic'
base_path = f'{PATH}/csv_database_energy_by_periods'
output_directory = f'../figures_chapter_5/pdfs/'

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

def plot_ridge_group(systems_energetics, group_name, terms_prefix, output_directory, special_case=None):
    """
    Plots Ridge plots for the specified group of energetic terms with optional value capping.

    Parameters:
    - systems_energetics: Dictionary of DataFrames with system data.
    - group_name: Name of the group for labeling purposes.
    - terms_prefix: Prefixes of terms to include in the plot.
    - output_directory: Directory to save the plot.
    - value_cap: Optional tuple (min_cap, max_cap) to cap values. None means no capping.
    """
    # Compute the value caps based on quantiles for the group
    value_cap = compute_group_caps(systems_energetics, terms_prefix, special_case=special_case)

    # Concatenate all systems' dataframes while retaining the system id and period
    all_data = pd.concat([df.assign(system_id=system_id) for system_id, df in systems_energetics.items()])

    # Filter columns by the specified term prefixes
    columns_to_keep = ['system_id', 'Unnamed: 0'] + [col for col in all_data.columns if col.startswith(tuple(terms_prefix))]
    all_data = all_data[columns_to_keep]

    # Melt the dataframe for plotting
    all_data_melted = all_data.melt(id_vars=['system_id', 'Unnamed: 0'], var_name='Term', value_name='Value')

    # Apply value capping if specified
    if value_cap is not None:
        min_cap, max_cap = value_cap
        all_data_melted['Value'] = all_data_melted['Value'].clip(lower=min_cap, upper=max_cap)

    # Create subplots
    terms = [col for col in all_data.columns if col.startswith(tuple(terms_prefix))]
    n_terms = len(terms)
    n_cols = 2
    n_rows = (n_terms + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=False)

    for idx, (ax, term) in enumerate(zip(axes.flatten(), terms)):
        sns.kdeplot(data=all_data_melted[all_data_melted['Term'] == term], x='Value', ax=ax, fill=True)
        ax.set_title(term.replace('(finite diff.)', ''), fontsize=TITLE_FONT_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
        ax.set_ylabel('Density', fontsize=LABEL_FONT_SIZE)
        ax.set_xlabel('Value', fontsize=LABEL_FONT_SIZE)
         # Add letter label
        ax.text(0.02, 0.93, f'({chr(65 + idx)})', transform=ax.transAxes, fontsize=TITLE_FONT_SIZE, fontweight='bold')

        if 'Energy' not in group_name:
            ax.axvline(x=0, color='black', linestyle='--')

    plt.tight_layout()
    plot_filename = f'ridge_plot_{group_name.replace(" ", "_").replace("/", "_")}_total.png'
    plot_path = os.path.join(output_directory, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved {plot_filename} in {output_directory}")

def plot_ridge_group_phases(systems_energetics, group_name, terms_prefix, output_directory, special_case=None):
    value_cap = compute_group_caps(systems_energetics, terms_prefix, special_case=special_case)
    all_data = pd.concat([df.assign(system_id=system_id) for system_id, df in systems_energetics.items()])
    all_data.rename(columns={'Unnamed: 0': 'Phase'}, inplace=True)

    relevant_columns = ['system_id', 'Phase'] + [col for col in all_data.columns if col.startswith(tuple(terms_prefix))]
    all_data = all_data[relevant_columns]

    # Exclude "residual" phase
    all_data = all_data[all_data['Phase'] != 'residual']

    terms = [col for col in all_data.columns if col.startswith(tuple(terms_prefix))]
    n_terms = len(terms)
    n_cols = 2
    n_rows = (n_terms + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)

    for idx, (ax, term) in enumerate(zip(axes.flatten(), terms)):
        for phase, color in COLOR_PHASES.items():
            if phase in ['residual', 'Total']:
                continue  # Skip "residual" and handle "Total" separately

            phase_data = all_data.loc[all_data['Phase'] == phase].copy()
            phase_data.loc[:, term] = phase_data[term].clip(lower=value_cap[0], upper=value_cap[1])

            if not phase_data.empty:
                sns.kdeplot(data=phase_data, x=term, ax=ax, fill=True, alpha=0.5, color=color, label=phase, bw_adjust=0.5)

        ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
        ax.set_ylabel('Density', fontsize=LABEL_FONT_SIZE)
        ax.set_xlabel('Value', fontsize=LABEL_FONT_SIZE)
        ax.set_title(term.replace('(finite diff.)', ''), fontsize=TITLE_FONT_SIZE)
        # Add letter label
        ax.text(0.02, 0.93, f'({chr(65 + idx)})', transform=ax.transAxes, fontsize=TITLE_FONT_SIZE, fontweight='bold')
    
        if 'Energy' not in group_name:
            ax.axvline(x=0, color='black', linestyle='--')

    plt.tight_layout()

    plot_filename = f'ridge_plot_{group_name.replace("/", "_")}.png'.replace('/', '_')  # Remove slashes from filenames
    plot_path = os.path.join(output_directory, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved {plot_filename} in {output_directory}")

if __name__ == "__main__":
    os.makedirs(output_directory, exist_ok=True)

    systems_energetics = read_life_cycles(base_path)

    # Define term prefixes for each group
    groups = {
        'Energy Terms': ['A', 'K'],
        'Conversion Terms': ['C'],
        'Boundary Terms': ['B'],
        'Generation/Dissipation Terms': ['G', 'R'],
        'Budgets': ['∂']
    }

    for group_name, terms_prefix in groups.items():
        special_case = None
        if group_name == 'Energy Terms':
            special_case = group_name
        plot_ridge_group(systems_energetics, group_name, terms_prefix, output_directory, special_case=special_case)
        plot_ridge_group_phases(systems_energetics, group_name, terms_prefix, output_directory, special_case=special_case)
