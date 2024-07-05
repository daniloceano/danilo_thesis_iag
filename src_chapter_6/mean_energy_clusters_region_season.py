import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

PHASES = ['incipient', 'intensification', 'mature', 'decay']
TERMS = ['Ck', 'Ca', 'Ke', 'Ge', 'BAe', 'BKe']
SEASONS = ['DJF', 'JJA']
REGIONS = ['SE-BR', 'LA-PLATA', 'ARG']

def read_patterns(results_path, PHASES, TERMS):
    """
    Read the energetics data for patterns from a JSON file.

    Args:
        results_path (str): The path to the directory containing the JSON file.
        PHASES (list): A list of strings representing the phases.
        TERMS (list): A list of strings representing the terms.

    Returns:
        list: A list of pandas DataFrames, each representing the energetics data for a pattern.
    """
    # Read the energetics data for patterns
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

def plot_combined_bar_plots(regions_seasons_data):
    """
    Plot the mean centroids values for each region and season in a combined bar plot.

    Args:
        regions_seasons_data (dict): A dictionary containing the data for each region and season.
    """
    mean_values = {region: {season: [] for season in SEASONS} for region in REGIONS}
    
    for region in regions_seasons_data:
        for season in regions_seasons_data[region]:
            for term in TERMS:
                term_values = [df[term].mean() for df in regions_seasons_data[region][season]]
                mean_values[region][season].append(np.mean(term_values))

    bar_width = 0.15
    x = np.arange(len(TERMS))
    fig, ax = plt.subplots(figsize=(15, 8))

    for i, region in enumerate(REGIONS):
        for j, season in enumerate(SEASONS):
            ax.bar(x + (i * len(SEASONS) + j) * bar_width, mean_values[region][season], bar_width,
                   label=f'{region} {season}')

    ax.set_xlabel('Terms')
    ax.set_ylabel('Mean Values')
    ax.set_title('Mean Centroids for Different Regions and Seasons')
    ax.set_xticks(x + bar_width * 2.5)
    ax.set_xticklabels(TERMS)
    ax.legend()

    plt.savefig('../figures_chapter_6/combined_bar_plot.png', dpi=300)

results_path = "../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic/results_kmeans/"

regions_seasons_data = {}
for season in SEASONS:
    for region in REGIONS:
        # Read the energetics data for patterns
        patterns_clusters_paths = f"{results_path}/{region}_{season}/IcItMD/"
        patterns_energetics, clusters_center, results = read_patterns(patterns_clusters_paths, PHASES, TERMS)

        if region not in regions_seasons_data:
            regions_seasons_data[region] = {}
        regions_seasons_data[region][season] = patterns_energetics

plot_combined_bar_plots(regions_seasons_data)
