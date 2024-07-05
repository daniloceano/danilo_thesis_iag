# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    eps_seasonality.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/07/05 12:40:10 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/05 12:40:12 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import pandas as pd
import numpy as np
from glob import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statannotations.Annotator import Annotator

PHASES = ['incipient', 'intensification', 'mature', 'decay']
TERMS = ['Ck', 'Ca', 'Ke', 'Ge', 'BAe', 'BKe']
SEASONS = ['DJF', 'JJA']
REGIONS = ['SE-BR', 'LA-PLATA', 'ARG']

COLOR_SEASONS = {
    'JJA': '#65a1e6',
    'MAM': '#f7b538',
    'DJF': '#d62828',
    'SON': '#9aa981'}

def read_patterns(results_path, PHASES, TERMS):
    patterns_json = glob(f'{results_path}/kmeans_results*.json')
    results = pd.read_json(patterns_json[0])
    clusters_center = results.loc['Cluster Center']
    patterns_energetics = []
    for i in range(len(clusters_center)):
        cluster_center = np.array(clusters_center.iloc[i])
        cluster_array = np.array(cluster_center).reshape(len(TERMS), len(PHASES))
        df = pd.DataFrame(cluster_array, columns=PHASES, index=TERMS).T
        df['Ke'] = df['Ke'] / 1e6  # Adjust the magnitude of Ke
        patterns_energetics.append(df)
    return patterns_energetics, clusters_center, results

def process_file(file_path):
    data = pd.read_csv(file_path)
    return pd.to_datetime(data['date'].loc[0])

def parallel_file_processing(file_paths):
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, file_paths), total=len(file_paths)))
    return results

def map_month_to_season(month):
    if month in [12, 1, 2]:
        return 'DJF'
    elif month in [3, 4, 5]:
        return 'MAM'
    elif month in [6, 7, 8]:
        return 'JJA'
    elif month in [9, 10, 11]:
        return 'SON'
    else:
        return 'Unknown'

def plot_seasonality(genesis_clusters):
    seasonality_data = []

    for cluster, dates in genesis_clusters.items():
        seasons = [map_month_to_season(date.month) for date in dates]
        season_counts = pd.Series(seasons).value_counts(normalize=True) * 100
        for season, percentage in season_counts.items():
            ep = cluster.replace('Cluster', 'EP')
            seasonality_data.append({'Cluster': ep, 'Season': season, 'Percentage': percentage})
    
    df_seasonality = pd.DataFrame(seasonality_data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Cluster', y='Percentage', hue='Season', data=df_seasonality, palette=COLOR_SEASONS, alpha=0.8)
    plt.ylabel('Percentage (%)', fontsize=16)
    plt.xlabel('', fontsize=16)
    plt.legend(fontsize=16, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.savefig('../figures_chapter_6/lps_seasonality.png', dpi=300)

def main():
    patterns_clusters_path = "../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic/results_kmeans/all_systems/IcItMD"
    patterns_energetics, clusters_center, results = read_patterns(patterns_clusters_path, PHASES, TERMS)
    ids_clusters = results.loc['Cyclone IDs']
    
    # Fetch CSV files with LPS data
    directory_path = '../results_chapter_5/database_tracks/'
    csv_files = glob(os.path.join(directory_path, '*.csv'))

    # Filter csv files for ids in ids_clusters
    csv_files_clusters = {}
    genesis_clusters = {}
    for cluster in ids_clusters.index:
        csv_files_clusters[cluster] = [file for file in csv_files if int(file.split('/')[-1].split('.')[0].split('_')[-1]) in ids_clusters.loc[cluster]]
        # Process files in parallel
        genesis_clusters[cluster] = parallel_file_processing(csv_files_clusters[cluster])
    
    # Plot the seasonality of each cluster
    plot_seasonality(genesis_clusters)

if __name__ == '__main__':
    main()
