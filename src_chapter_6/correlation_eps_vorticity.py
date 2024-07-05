import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from scipy import stats
from statannotations.Annotator import Annotator

PHASES = ['incipient', 'intensification', 'mature', 'decay']
TERMS = ['Ck', 'Ca', 'Ke', 'Ge', 'BAe', 'BKe']
SEASONS = ['DJF', 'JJA']
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
    vor42_values = pd.to_numeric(data['vor 42'], errors='coerce').dropna().values if 'vor 42' in data.columns else []
    return vor42_values

def parallel_file_processing(file_paths):
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, file_paths), total=len(file_paths)))
    return results

def main():
    patterns_clusters_path = "../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic/results_kmeans/all_systems/IcItMD"
    patterns_energetics, clusters_center, results = read_patterns(patterns_clusters_path, PHASES, TERMS)
    ids_clusters = results.loc['Cyclone IDs']
    
    # Fetch CSV files with LPS data
    directory_path = '../results_chapter_5/database_tracks/'
    csv_files = glob(os.path.join(directory_path, '*.csv'))

    # Filter csv files for ids in ids_clusters
    csv_files_clusters = {}
    vorticity_clusters = {}
    for cluster in ids_clusters.index:
        csv_files_clusters[cluster] = [file for file in csv_files if int(file.split('/')[-1].split('.')[0].split('_')[-1]) in ids_clusters.loc[cluster]]
        # Process files in parallel
        vorticity_clusters[cluster] = parallel_file_processing(csv_files_clusters[cluster])

    # Prepare data for visualization
    all_data = []
    for cluster, vorticities in vorticity_clusters.items():
        for vorticity_list in vorticities:
            for vor in vorticity_list:
                all_data.append({'EP': cluster, 'Vorticity': vor})

    df = pd.DataFrame(all_data)
    
    # Perform Kruskal-Wallis test
    clusters = df['EP'].unique()
    data_by_cluster = [df[df['EP'] == cluster]['Vorticity'] for cluster in clusters]
    h_statistic, p_value = stats.kruskal(*data_by_cluster)
    print(f'Kruskal-Wallis H-statistic: {h_statistic}, p-value: {p_value}')
    
    # Perform pairwise comparisons if Kruskal-Wallis test is significant
    if p_value < 0.05:
        order = clusters.tolist()
        pairs = [(clusters[i], clusters[j]) for i in range(len(clusters)) for j in range(i+1, len(clusters))]
        
        # Visualization
        plt.figure(figsize=(10, 10))
        x, y = 'EP', 'Vorticity'
        ax = sns.boxplot(x=x, y=y, data=df, palette='pastel')
        # Annotate the boxplot with pairwise comparisons
        annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
        annotator.apply_and_annotate()

        ax.set_xlabel('')
        ax.set_ylabel('Relative Vorticity ($-10^{-5}$ $s^{-1}$)', fontsize=16)

        # Increase x and y-axis tick labels font size
        ax.yaxis.set_tick_params(labelsize=14)
        ax.xaxis.set_tick_params(labelsize=14)
        
        plt.tight_layout()
        plt.savefig('../figures_chapter_6/boxplot_vorticity_by_cluster.png', dpi=300)
    else:
        print("No significant differences found among clusters with Kruskal-Wallis test.")

if __name__ == '__main__':
    main()
