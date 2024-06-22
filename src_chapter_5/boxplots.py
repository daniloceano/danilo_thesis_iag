# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    boxplots.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/03/02 17:31:28 by daniloceano       #+#    #+#              #
#    Updated: 2024/06/22 16:35:20 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

PATH = '/Users/danilocoutodesouza/Documents/Programs_and_scripts/energetic_patterns_cyclones_south_atlantic'
base_path = f'{PATH}/csv_database_energy_by_periods'
output_directory = f'../figures_chapter_5/boxplots/'

COLOR_PHASES = {
    'Total': 'grey',
    'incipient': '#65a1e6',
    'intensification': '#f7b538',
    'mature': '#d62828',
    'decay': '#9aa981',
    'intensification 2': '#ca6702',
    'mature 2': '#9b2226',
    'decay 2': '#386641',
}

COLORS_TERMS = ['#3B95BF', '#87BF4B', '#BFAB37', '#BF3D3B', '#873e23', '#A13BF0']

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
                df = df.rename(columns={'Unnamed: 0': 'phase'})
                df.index = range(1, len(df) + 1)
                systems_energetics[system_id] = df
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return systems_energetics

def compute_total_phase(systems_energetics):
    """
    Computes the mean values across all phases for each system to represent the "Total" phase.
    """
    for system_id, df in systems_energetics.items():
        mean_values = df.mean(numeric_only=True)
        mean_values['phase'] = 'Total'
        systems_energetics[system_id] = pd.concat([df, pd.DataFrame([mean_values])], ignore_index=True)
    return systems_energetics

def plot_box_plots_total(systems_energetics, groups, output_directory):
    """
    Generates a single figure with subplots for each group of energetic terms for the Total phase.
    """
    all_data = pd.concat(systems_energetics.values(), ignore_index=True)

    n_groups = len(groups)
    n_cols = 2
    n_rows = (n_groups + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 7 * n_rows), dpi=300)

    for idx, (ax, (group_name, terms_prefix)) in enumerate(zip(axes.flatten(), groups.items())):
        terms = [col for col in all_data.columns if col.startswith(tuple(terms_prefix))]
        
        # Filter data for the Total phase
        total_phase_data = all_data[all_data['phase'] == 'Total']
        
        # Melt the DataFrame to have a long format suitable for seaborn
        melted_data = total_phase_data.melt(id_vars=['phase'], value_vars=terms, var_name='Term', value_name='Value')
        
        # Remove "(finite diff.)" from budget terms
        if 'Budgets' in group_name:
            melted_data['Term'] = melted_data['Term'].str.replace(r' \(finite diff\.\)', '', regex=True)
        
        sns.boxplot(x='Term', y='Value', data=melted_data, palette=COLORS_TERMS, hue='Term', ax=ax)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.8, linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE + 2)
        ax.set_xlabel('')
        ax.set_ylabel('')
        # Add letter label
        ax.text(0.02, 0.95, f'({chr(65 + idx)})', transform=ax.transAxes, fontsize=TITLE_FONT_SIZE, fontweight='bold')

    # Remove any empty subplots
    if n_groups < n_rows * n_cols:
        for idx in range(n_groups, n_rows * n_cols):
            fig.delaxes(axes.flatten()[idx])

    plt.tight_layout()
    os.makedirs(output_directory, exist_ok=True)
    plot_filename = 'box_plot_Total_all_groups.png'
    plot_path = os.path.join(output_directory, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved {plot_filename} in {output_directory}")

def plot_box_plots_by_phase(systems_energetics, group_name, terms_prefix, output_directory):
    """
    Generates a figure for each group of terms with subplots for each term, and each subplot contains multiple box plots for each phase.
    """
    all_data = pd.concat(systems_energetics.values(), ignore_index=True)
    all_data = all_data[all_data['phase'] != 'Total']
    terms = [col for col in all_data.columns if col.startswith(tuple(terms_prefix))]

    n_terms = len(terms)
    n_cols = 2
    n_rows = (n_terms + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)

    for idx, (ax, term) in enumerate(zip(axes.flatten(), terms)):
        sns.boxplot(x='phase', y=term, data=all_data, palette=COLOR_PHASES.values(), hue='phase',
                    order=['incipient', 'intensification', 'mature', 'decay', 'intensification 2', 'mature 2', 'decay 2'], ax=ax)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.8, linewidth=0.5)
        ax.set_title(term, fontsize=TITLE_FONT_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
        ax.tick_params(axis='x', which='major', labelrotation=45)
        # Add letter label
        ax.text(0.02, 0.93, f'({chr(65 + idx)})', transform=ax.transAxes, fontsize=TITLE_FONT_SIZE, fontweight='bold')

    plt.tight_layout()
    plot_filename = f'box_plot_{group_name.replace(" ", "_").replace("/", "_")}.png'
    plot_path = os.path.join(output_directory, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved {plot_filename} in {output_directory}")

if __name__ == "__main__":
    os.makedirs(output_directory, exist_ok=True)

    systems_energetics = read_life_cycles(base_path)
    systems_energetics = compute_total_phase(systems_energetics)

    # Define term prefixes for each group
    groups = {
        'Energy Terms': ['A', 'K'],
        'Conversion Terms': ['C'],
        'Boundary Terms': ['B'],
        'Generation/Dissipation Terms': ['G', 'R'],
        'Budgets': ['∂']
    }

    plot_box_plots_total(systems_energetics, groups, output_directory)

    for group_name, terms_prefix in groups.items():
        plot_box_plots_by_phase(systems_energetics, group_name, terms_prefix, output_directory)