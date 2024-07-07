# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    eps_genesis_regions.py                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/07/07 12:40:10 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/07 11:41:47 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

REGIONS = ['SE-BR', 'LA-PLATA', 'ARG']
RESULTS_PATH = '../results_chapter_5'
PATTERNS_PATH = "../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic/results_kmeans/all_systems/IcItMD"

COLOR_REGIONS = {
    'ARG': '#65a1e6',
    'SE-BR': '#d62828',
    'LA-PLATA': '#9aa981'}

def read_track_ids(region_files):
    track_ids = {}
    for region, file_path in region_files.items():
        data = pd.read_csv(file_path)
        track_ids[region] = data['track_id'].tolist()
    return track_ids

def read_patterns(patterns_path):
    patterns_json = glob(os.path.join(patterns_path, 'kmeans_results*.json'))
    results = pd.read_json(patterns_json[0])
    ids_clusters = results.loc['Cyclone IDs']
    return ids_clusters

def plot_genesis_region_counts(track_ids):
    total_systems = sum(len(ids) for ids in track_ids.values())
    genesis_percentages = {region: (len(ids) / total_systems) * 100 for region, ids in track_ids.items()}
    
    df_genesis_counts = pd.DataFrame(list(genesis_percentages.items()), columns=['Region', 'Percentage'])

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Region', y='Percentage', data=df_genesis_counts, palette=COLOR_REGIONS, alpha=0.8)
    plt.ylabel('Percentage of Systems (%)', fontsize=16)
    plt.title(f'Percentage of Systems by Genesis Region (Total: {total_systems} Systems)', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.savefig('../figures_chapter_6/genesis_region_percentages.png', dpi=300)

def plot_ep_frequencies(ids_clusters, track_ids):
    ep_frequencies = []

    for region, ids in track_ids.items():
        for cluster in ids_clusters.index:
            ep = cluster.replace('Cluster', 'EP')
            count = len(set(ids).intersection(set(ids_clusters.loc[cluster])))
            total_in_region = len(ids)
            percentage = (count / total_in_region) * 100 if total_in_region > 0 else 0
            ep_frequencies.append({'Region': region, 'EP': ep, 'Percentage': percentage, 'Count': count})
    
    df_ep_frequencies = pd.DataFrame(ep_frequencies)

    plt.figure(figsize=(12, 8))
    barplot = sns.barplot(x='Percentage', y='EP', hue='Region', data=df_ep_frequencies, palette=COLOR_REGIONS, alpha=0.8)
    plt.xlabel('Percentage of Systems (%)', fontsize=16)
    plt.title('Relative Frequency of EPs by Genesis Region', fontsize=18)
    handles, labels = barplot.get_legend_handles_labels()
    plt.legend(handles[:3], labels[:3], title='Region', fontsize=12, title_fontsize='13', loc='lower right')
    plt.tick_params(axis='both', which='major', labelsize=14)

    # Annotate each bar with the total count of systems
    for p, count in zip(barplot.patches, df_ep_frequencies['Count']):
        plt.text(p.get_width() + 0.5, p.get_y() + p.get_height() / 2, f'{int(count)}', ha='center', va='center', fontsize=12)

    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('../figures_chapter_6/ep_frequencies_by_region.png', dpi=300)
    plt.show()

def main():
    # File paths for each region's track ids
    region_files = {
        'SE-BR': os.path.join(RESULTS_PATH, 'track_ids_SE-BR.csv'),
        'LA-PLATA': os.path.join(RESULTS_PATH, 'track_ids_LA-PLATA.csv'),
        'ARG': os.path.join(RESULTS_PATH, 'track_ids_ARG.csv')
    }

    # Read track ids from the CSV files
    track_ids = read_track_ids(region_files)

    # Read EP patterns
    ids_clusters = read_patterns(PATTERNS_PATH)

    # Plot the percentages of systems by genesis region
    plot_genesis_region_counts(track_ids)

    # Plot the relative frequency of EPs by genesis region
    plot_ep_frequencies(ids_clusters, track_ids)

if __name__ == '__main__':
    main()
