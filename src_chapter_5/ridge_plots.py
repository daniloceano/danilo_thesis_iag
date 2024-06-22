# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ridge_plots.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/02/29 14:56:47 by daniloceano       #+#    #+#              #
#    Updated: 2024/06/22 20:08:04 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

PATH = '/Users/danilocoutodesouza/Documents/Programs_and_scripts/energetic_patterns_cyclones_south_atlantic'
base_path = f'{PATH}/csv_database_energy_by_periods'
output_directory = f'../figures_chapter_5/ridge_plots/'

COLOR_PHASES = {
    'incipient': '#87b3e3',
    'intensification': '#f9c87d',
    'intensification 2': '#d67e37',
    'mature': '#de6a6a',
    'mature 2': '#a54c29',
    'decay': '#a6b38f',
    'decay 2': '#4a7b4f',
}

PHASE_MAPPING = {
    'incipient': 'Ic',
    'intensification': 'It',
    'intensification 2': 'It2',
    'mature': 'M',
    'mature 2': 'M2',
    'decay': 'D',
    'decay 2': 'D2',
}


LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 12
TITLE_FONT_SIZE = 16
LEGEND_FONT_SIZE = 12

def read_life_cycles(base_path):
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
    all_values = []
    for df in systems_energetics.values():
        relevant_cols = [col for col in df.columns if col.startswith(tuple(terms_prefix))]
        all_values.extend(df[relevant_cols].values.flatten())
    all_values = pd.Series(all_values).dropna()
    if special_case == 'Energy Terms':
        min_cap = 0
        max_cap = all_values.quantile(0.8)
    else:
        q2 = all_values.quantile(0.2)
        q8 = all_values.quantile(0.8)
        highest_cap = np.amax(np.abs([q2, q8]))
        min_cap, max_cap = -highest_cap, highest_cap
    return min_cap, max_cap

def plot_ridge_group_facet(systems_energetics, group_name, terms_prefix, output_directory, special_case=None):
    value_cap = compute_group_caps(systems_energetics, terms_prefix, special_case=special_case)
    all_data = pd.concat([df.assign(system_id=system_id) for system_id, df in systems_energetics.items()])
    all_data.rename(columns={'Unnamed: 0': 'Phase'}, inplace=True)
    relevant_columns = ['system_id', 'Phase'] + [col for col in all_data.columns if col.startswith(tuple(terms_prefix))]
    all_data = all_data[relevant_columns]
    all_data = all_data[(all_data['Phase'] != 'residual')]
    all_data = all_data[(all_data['Phase'] != 'incipient 2')]
    terms = [col for col in all_data.columns if col.startswith(tuple(terms_prefix))]
    for term in terms:
        phase_data = all_data[['system_id', 'Phase', term]].copy()
        phase_data[term] = phase_data[term].clip(lower=value_cap[0], upper=value_cap[1])
        phase_data_melted = phase_data.melt(id_vars=['system_id', 'Phase'], var_name='Term', value_name='Value')
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), 'axes.linewidth': 2})
        palette = [COLOR_PHASES[phase] for phase in all_data['Phase'].unique()]
        order = ['incipient', 'intensification', 'mature', 'decay', 'intensification 2', 'mature 2', 'decay 2']
        g = sns.FacetGrid(phase_data_melted, row="Phase", hue="Phase", aspect=9, height=1.2, palette=palette, row_order=order)
        g.map_dataframe(sns.kdeplot, x="Value", fill=True, alpha=1)
        g.map_dataframe(sns.kdeplot, x="Value", color='black')
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, PHASE_MAPPING[label], color='black', fontsize=24, ha="left", va="center", transform=ax.transAxes)
        g.map(label, "Phase")

        for ax, phase in zip(g.axes.flatten(), order):
            phase_values = phase_data[phase_data['Phase'] == phase][term]
            mean_value = phase_values.mean()
            median_value = phase_values.median()
            ax.axvline(mean_value, ymax=0.8, color='k', linestyle='-')
            ax.axvline(median_value, ymax=0.8, color='k', linestyle='--')
            if group_name != 'Energy Terms':
                ax.axvline(0, color='k', ymax=0.8, linestyle='-', alpha=0.5, linewidth=0.5) 
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis='x', which='minor', length=4, color='gray', size=16)
            ax.tick_params(axis='x', which='both', labelsize=16)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10)) 
            ax.set_xlabel(term.replace('(finite diff.)', ''), fontsize=20)     

        g.set_titles("")
        g.fig.subplots_adjust(hspace=-.5)
        g.set(yticks=[], ylabel="")
        g.despine(left=True)
        plot_filename = f'ridge_plot_{group_name.replace("/", "_")}_{term.replace("/", "_").replace("(finite diff.)", "")}.png'
        plot_path = os.path.join(output_directory, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved {plot_filename} in {output_directory}")

if __name__ == "__main__":
    os.makedirs(output_directory, exist_ok=True)
    systems_energetics = read_life_cycles(base_path)
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
        plot_ridge_group_facet(systems_energetics, group_name, terms_prefix, output_directory, special_case=special_case)
