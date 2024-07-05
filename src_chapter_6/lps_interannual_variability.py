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
REGIONS = ['SE-BR', 'LA-PLATA', 'ARG']

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

def plot_interannual_variability(genesis_clusters):
    interannual_data = []

    for cluster, dates in genesis_clusters.items():
        years = [date.year for date in dates]
        year_counts = pd.Series(years).value_counts(normalize=True) * 100
        for year, percentage in year_counts.items():
            ep = cluster.replace('Cluster', 'EP')
            interannual_data.append({'Cluster': ep, 'Year': year, 'Percentage': percentage})
    
    df_interannual = pd.DataFrame(interannual_data)
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Year', y='Percentage', hue='Cluster', data=df_interannual, palette='pastel', marker='o')
    
    plt.title('Interannual Variability of Cyclone Clusters', fontsize=18)
    plt.ylabel('Percentage (%)', fontsize=16)
    plt.xlabel('Year', fontsize=16)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True)
    plt.savefig('../figures_chapter_6/lps_interannual_variability.png', dpi=300)

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
    
    # Plot the interannual variability of each cluster
    plot_interannual_variability(genesis_clusters)

if __name__ == '__main__':
    main()
