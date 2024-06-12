import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DATABASE_DIR = '/Users/danilocoutodesouza/Documents/Programs_and_scripts/SWSA-cyclones_energetic-analysis/periods_species_statistics/70W-no-continental/periods_database'

regions = ['ARG', 'LA-PLATA', 'SE-BR']
seasons = ['DJF', 'JJA']

database = glob(f'{DATABASE_DIR}/*.csv')

# Define metrics to plot
metrics = ['Total Time (h)', 'Total Distance (km)', 'Mean Speed (m/s)',  
           'Mean Vorticity (−1 × 10−5 s−1)', 'Mean Growth Rate (10^−5 s^−1 day^-1)']

# Color mapping for regions (reversed colors for ARG and SE-BR)
COLOR_REGIONS = {
    'ARG': '#3e8fc1',
    'LA-PLATA': '#adad38',
    'SE-BR': '#d73027',
}

# Simplified name mapping for metrics
METRIC_NAMES = {
    'Total Time (h)': 'total_time',
    'Total Distance (km)': 'total_distance',
    'Mean Speed (m/s)': 'mean_speed',
    'Mean Vorticity (−1 × 10−5 s−1)': 'mean_vorticity',
    'Mean Growth Rate (10^−5 s^−1 day^-1)': 'mean_growth_rate'
}

# LaTeX formatted labels
LATEX_LABELS = {
    'Total Time (h)': 'Total Time (h)',
    'Mean Speed (m/s)': r'Mean Speed (m s$^{-1}$)',
    'Total Distance (km)': 'Total Distance (km)',
    'Mean Vorticity (−1 × 10−5 s−1)': r'Mean Vorticity ($-10^{-5}$ s$^{-1}$)',
    'Mean Growth Rate (10^−5 s^−1 day^-1)': r'Mean Growth Rate ($10^{-5}$ s$^{-1}$ day$^{-1}$)'
}

# Label order
LABELS = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)"]

# Output directory
output_directory = '../figures_chapter_4/'
os.makedirs(output_directory, exist_ok=True)

# Initialize an empty DataFrame to hold all data
all_data = pd.DataFrame()

# Read and concatenate all data into a single DataFrame
for file in database:
    df = pd.read_csv(file)
    all_data = pd.concat([all_data, df])

# Filter data to include only the specified regions
filtered_data = all_data[all_data['Genesis Region'].isin(regions)]

# Remove residual phases
filtered_data = filtered_data[~filtered_data['phase'].str.contains('residual')]

# Determine global x-axis limits for each metric
total_global_xlims = {}
for metric in metrics:
    total_metric_data = filtered_data[filtered_data['phase'] == 'Total'][metric]
    min_val = total_metric_data.min()
    max_val = total_metric_data.max()
    if metric != 'Mean Growth Rate (10^−5 s^−1 day^-1)':
        min_val = 0
    total_global_xlims[metric] = (min_val, max_val)

# Determine global y-axis limits for each metric
global_ylims = {}
for metric in metrics:
    max_y = 0
    for season in seasons:
        mean_data = filtered_data[(filtered_data['Genesis Season'] == season) & (filtered_data['phase'] == 'Total')]
        for region in regions:
            region_data = filtered_data[(filtered_data['Genesis Season'] == season) & 
                                        (filtered_data['Genesis Region'] == region) & 
                                        (filtered_data['phase'] == 'Total')]
            if not region_data.empty:
                y = sns.kdeplot(region_data[metric]).get_lines()[-1].get_ydata()
                kde = (y / np.sum(y) * 100)
                max_y = max(max_y, kde.max())
            if not mean_data.empty:
                y = sns.kdeplot(mean_data[metric]).get_lines()[-1].get_ydata()
                kde = (y / np.sum(y) * 100)
                max_y = max(max_y, kde.max())
    global_ylims[metric] = max_y

# Plot PDFs for the 'Total' phase separately with all metrics in a multi-panel plot
fig, axes = plt.subplots(len(metrics), len(seasons), figsize=(18, 18), sharey=False)

for i, metric in enumerate(metrics):
    for j, season in enumerate(seasons):
        ax = axes[i, j]
        
        # Plot mean across all regions in black
        mean_data = filtered_data[(filtered_data['Genesis Season'] == season) & (filtered_data['phase'] == 'Total')]
        kdeplot = sns.kdeplot(mean_data[metric], ax=ax, label='Mean All Regions', color='black', linewidth=3)
        
        # Normalize the y-axis to show percentages
        y = kdeplot.get_lines()[-1].get_ydata()
        kdeplot.get_lines()[-1].set_ydata(y / np.sum(y) * 100)
        
        mean_val = mean_data[metric].mean()
        std_val = mean_data[metric].std()
        ax.text(0.95, 0.92, 
                f'Mean: {mean_val:.2f}, ± {std_val:.2f}', 
                color='black', transform=ax.transAxes, ha='right', fontsize=12, fontweight='bold')

        for region in regions:
            region_data = filtered_data[(filtered_data['Genesis Season'] == season) & 
                                        (filtered_data['Genesis Region'] == region) & 
                                        (filtered_data['phase'] == 'Total')]
            
            kdeplot = sns.kdeplot(region_data[metric], ax=ax, label=region, color=COLOR_REGIONS[region], linewidth=3)
            
            # Normalize the y-axis to show percentages
            y = kdeplot.get_lines()[-1].get_ydata()
            kde = (y / np.sum(y) * 100)
            kdeplot.get_lines()[-1].set_ydata(kde)
            
            # Calculate mean and standard deviation
            mean_val = region_data[metric].mean()
            std_val = region_data[metric].std()
            ax.text(0.95, 0.92 - (0.1 * (regions.index(region) + 1)), 
                    f'{region}: {mean_val:.2f}, ± {std_val:.2f}', 
                    color=COLOR_REGIONS[region], transform=ax.transAxes, ha='right', fontsize=12, fontweight='bold')

        if i == 0:
            ax.set_title(f'{season}', fontsize=14, fontweight='bold')

        ax.text(0.01, 1.05, f'{LABELS[i * len(seasons) + j]}', transform=ax.transAxes, fontsize=14, fontweight='bold')

        # Set the xlabel with LaTeX formatting
        ax.set_xlabel(LATEX_LABELS[metric], fontsize=14)
        ax.set_ylabel('Percentage (%)', fontsize=14)
        ax.set_xlim(total_global_xlims[metric])
        ax.set_ylim(0, global_ylims[metric] + 0.1)
        ax.tick_params(axis='both', labelsize=12)

        # Add gridlines
        ax.grid(axis='both', linestyle='--', linewidth=1, alpha=0.5)

        # Add vertical line at 0 for 'Mean Growth Rate'
        if metric == 'Mean Growth Rate (10^−5 s^−1 day^-1)':
            ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

# Move legend to the right outside the plot
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='center left', ncol=7, bbox_to_anchor=(0.32, 0.025), prop={'size': 14})

plt.tight_layout()
plt.subplots_adjust(bottom=0.08)

filename = os.path.join(output_directory, 'pdf_total_phase_all_metrics.png')
plt.savefig(filename)
print(f'Saved {filename}')
plt.close()