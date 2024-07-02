import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from scipy import stats

# Define the directory containing the CSV files
directory_path = '../results_chapter_5/database_tracks/'

# Function to read and process each CSV file
def process_file(file_path):
    data = pd.read_csv(file_path)
    ca_values = pd.to_numeric(data['Ca'], errors='coerce').dropna().values if 'Ca' in data.columns else []
    ce_values = pd.to_numeric(data['Ce'], errors='coerce').dropna().values if 'Ce' in data.columns else []
    ke_values = pd.to_numeric(data['Ke'], errors='coerce').dropna().values if 'Ke' in data.columns else []
    vor42_values = pd.to_numeric(data['vor 42'], errors='coerce').dropna().values if 'vor 42' in data.columns else []
    return ca_values, ce_values, ke_values, vor42_values

# Function to parallelize file processing
def parallel_file_processing(file_paths):
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, file_paths), total=len(file_paths)))
    return results

# Function to merge processed data
def merge_processed_data(results):
    Ca_values, Ce_values, Ke_values, Vor42_values = [], [], [], []
    for ca_values, ce_values, ke_values, vor42_values in results:
        Ca_values.extend(ca_values)
        Ce_values.extend(ce_values)
        Ke_values.extend(ke_values)
        Vor42_values.extend(vor42_values)
    return (np.array(Ca_values), np.array(Ce_values), np.array(Ke_values) / 1e6, np.array(Vor42_values))

# Function to calculate Pearson correlation
def calculate_pearson_correlation(dataframe):
    return dataframe.corr().iloc[0, 1]

# Function to create and save joint plots
def create_jointplot(x, y, data, x_label, y_label, xlim, ylim, filename, regression=True, density_label='Density'):
    pr = lambda a, b: stats.pearsonr(a, b)[0]
    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))

    g = sns.jointplot(x=x, y=y, data=data, kind='kde', fill=True, cmap="rainbow", space=1, cbar=True,
                      n_levels=10, cbar_kws={"format": formatter, "label": density_label}, 
                      stat_func=pr, annot_kws={'stat':'pearsonr'})

    if regression:
        sns.regplot(x=x, y=y, data=data, ax=g.ax_joint, scatter=False, color='r', label=f'r = {calculate_pearson_correlation(data):.2f}')
        g.ax_joint.legend(loc='best')

    g.ax_joint.set_xlim(xlim)
    g.ax_joint.set_ylim(ylim)
    g.ax_joint.set_xlabel(x_label)
    g.ax_joint.set_ylabel(y_label)

    plt.savefig(f"{filename}.png", dpi=300)
    print(f"Saved {filename}.png")

# Main function
def main():
    # Fetch CSV files
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))

    # Process files in parallel
    results = parallel_file_processing(csv_files)

    # Merge processed data
    Ca_values, Ce_values, Ke_values, Vor42_values = merge_processed_data(results)

    # Create DataFrames with the combined data
    combined_data_ca_ce = pd.DataFrame({'Ca': Ca_values, 'Ce': Ce_values})
    combined_data_ke_vor42 = pd.DataFrame({'Ke': Ke_values, 'vor42': Vor42_values})

    # Create and save joint plots
    create_jointplot('Ca', 'Ce', combined_data_ca_ce, 'Ca (W$^{-2}$)', 'Ce (W$^{-2}$)', 
                     [combined_data_ca_ce['Ca'].min() * 0.6, combined_data_ca_ce['Ca'].max() * 0.6], 
                     [combined_data_ca_ce['Ce'].min() * 0.6, combined_data_ca_ce['Ce'].max() * 0.6], 
                     '../figures_chapter_6/correlation_ca_ce')

    create_jointplot('Ke', 'vor42', combined_data_ke_vor42, 'Eddy Kinetic Energy ($10^6$ J m$^{-2}$)', 'Relative Vorticity ($10^{-5}$ s$^{-1}$)', 
                     [0, combined_data_ke_vor42['Ke'].max() * 0.5], 
                     [combined_data_ke_vor42['vor42'].min(), combined_data_ke_vor42['vor42'].max() * 0.8], 
                     '../figures_chapter_6/correlation_ke_vor42')

if __name__ == '__main__':
    main()
