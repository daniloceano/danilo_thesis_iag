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
    if 'Ca' in data.columns and 'Ce' in data.columns:
        ca_values = pd.to_numeric(data['Ca'], errors='coerce').dropna().values
        ce_values = pd.to_numeric(data['Ce'], errors='coerce').dropna().values
        return ca_values, ce_values
    return [], []

def main():
    # Initialize lists to store values
    Ca_values = []
    Ce_values = []

    # Use ProcessPoolExecutor to parallelize file processing with a progress bar
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, csv_files), total=len(csv_files)))

    # Collect results
    for ca_values, ce_values in results:
        Ca_values.extend(ca_values)
        Ce_values.extend(ce_values)

    # Convert lists to numpy arrays
    Ca_values = np.array(Ca_values)
    Ce_values = np.array(Ce_values)

    # Create a DataFrame with the combined data
    combined_data = pd.DataFrame({'Ca': Ca_values, 'Ce': Ce_values})

    # Calculate Pearson correlation coefficient
    pearson_corr = combined_data.corr().iloc[0, 1]

    pr = lambda a, b: stats.pearsonr(a, b)[0]

    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))

    # Create the correlation plot using Seaborn
    g = sns.jointplot(x='Ca', y='Ce', data=combined_data, kind='kde', fill=True, cmap="rainbow", space=1, cbar=True,
                      n_levels=10, cbar_kws={"format": formatter, "label": 'Density'},
                      stat_func=pr, annot_kws={'stat':'pearsonr'})
    
    # Set axis limits
    x0, x1 = g.ax_joint.get_xlim()
    y0, y1 = g.ax_joint.get_ylim()
    g.ax_joint.set_xlim([x0 * 0.6, x1 * 0.6])
    g.ax_joint.set_ylim([y0 * 0.6, y1 * 0.6])

    # Add a regression line
    sns.regplot(x='Ca', y='Ce', data=combined_data, ax=g.ax_joint, scatter=False, color='r', label=f'r = {pearson_corr:.2f}')


    # Add a line x=y
    lims = [max(x0, y0), min(x1, y1)]
    g.ax_joint.plot(lims, lims, '-k', label='y = x')

    # Add a legend
    g.ax_joint.legend(loc='best')



    fname = '../figures_chapter_6/correlation_ca_ce'
    plt.savefig(f"{fname}.png", dpi=300)
    print(f"Saved {fname}.png")

if __name__ == '__main__':
    main()
