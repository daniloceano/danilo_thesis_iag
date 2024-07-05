import os
import pandas as pd
import numpy as np
from glob import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
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
        month_years = [date.strftime('%Y-%m') for date in dates]  # Extract year and month
        month_year_counts = pd.Series(month_years).value_counts().sort_index()
        for month_year, count in month_year_counts.items():
            ep = cluster.replace('Cluster', 'EP')
            interannual_data.append({'Cluster': ep, 'MonthYear': month_year, 'Count': count})
    
    df_interannual = pd.DataFrame(interannual_data)
    df_interannual['MonthYear'] = pd.to_datetime(df_interannual['MonthYear'])
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='MonthYear', y='Count', hue='Cluster', data=df_interannual, palette='pastel', marker='o')
    plt.title('Monthly Interannual Variability of Cyclone Clusters')
    plt.ylabel('Count', fontsize=16)
    plt.xlabel('Month-Year', fontsize=16)
    plt.legend(title='Cluster', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True)
    plt.savefig('../figures_chapter_6/lps_interannual_variability_monthly.png', dpi=300)
    
    # Trend analysis for each cluster
    for cluster in df_interannual['Cluster'].unique():
        cluster_data = df_interannual[df_interannual['Cluster'] == cluster]
        cluster_data = cluster_data.set_index('MonthYear').resample('M').sum().fillna(0)  # Ensure continuous time series
        
        # Decompose the time series
        stl = STL(cluster_data['Count'], seasonal=13)
        result = stl.fit()
        
        # Extract the trend component
        trend = result.trend
        
        # Perform linear regression on the trend component
        X = np.arange(len(trend)).reshape(-1, 1)  # Time variable
        y = trend.values
        
        # Add constant for intercept
        X = sm.add_constant(X)
        
        # Fit linear regression model
        model = sm.OLS(y, X).fit()
        
        # Print the summary of the model
        print(f'Trend analysis for {cluster}:')
        print(model.summary())
        
        # Plot the trend line
        cluster_data['Trend'] = model.predict(X)
        plt.figure(figsize=(12, 6))
        plt.plot(cluster_data.index, cluster_data['Count'], label='Monthly Counts')
        plt.plot(cluster_data.index, cluster_data['Trend'], label='Trend Line', color='red')
        plt.title(f'Trend Analysis for {cluster}')
        plt.ylabel('Monthly Count')
        plt.xlabel('Month-Year')
        plt.legend()
        plt.grid(True)
        plt.show()

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
