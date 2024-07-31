import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATABASE_DIR = '/Users/danilocoutodesouza/Documents/Programs_and_scripts/SWSA-cyclones_energetic-analysis/periods_species_statistics/70W-no-continental/periods_database'

regions = ['ARG', 'LA-PLATA', 'SE-BR']
seasons = ['DJF', 'JJA']

database = glob(f'{DATABASE_DIR}/*.csv')

# Define metrics to plot
metrics = ['Total Time (h)', 'Total Distance (km)', 'Mean Speed (m/s)',  
           'Mean Vorticity (−1 × 10−5 s−1)', 'Mean Growth Rate (10^−5 s^−1 day^-1)']

# Color mapping for phases
COLOR_PHASES = {
    'Total': '#1d3557',
    'incipient': '#65a1e6',
    'intensification': '#f7b538',
    'intensification 2': '#ca6702',
    'mature': '#d62828',
    'mature 2': '#9b2226',
    'decay': '#9aa981',
    'decay 2': '#386641',
}

# Simplified name mapping for metrics
METRIC_NAMES = {
    'Total Time (h)': 'total_time',
    'Total Distance (km)': 'total_distance',
    'Mean Speed (m/s)': 'mean_speed',
    'Mean Vorticity (−1 × 10−5 s−1)': 'mean_vorticity',
    'Mean Growth Rate (10^−5 s^−1 day^-1)': 'mean_growth_rate'
}

METRIC_LIMITS = {
    'Total Time (h)': (0, 80),
    'Total Distance (km)': (0, 4000),
    'Mean Speed (m/s)': (0, 22),
    'Mean Vorticity (−1 × 10−5 s−1)': (0, 8),
    'Mean Growth Rate (10^−5 s^−1 day^-1)': (-2.5, 3.5)
}

# LaTeX formatted labels
LATEX_LABELS = {
    'Total Time (h)': 'Total Time (h)',
    'Mean Speed (m/s)': r'Mean Speed (m s$^{-1}$)',
    'Total Distance (km)': 'Total Distance (km)',
    'Mean Vorticity (−1 × 10−5 s−1)': 'Mean Vorticity\n($-10^{-5}$ s$^{-1}$)',
    'Mean Growth Rate (10^−5 s^−1 day^-1)': 'Mean Growth Rate\n($10^{-5}$ s$^{-1}$ day$^{-1}$)'
}

PHASE_ORDER = ['incipient', 'intensification', 'mature', 'decay', 'intensification 2', 'mature 2', 'decay 2']

output_directory = '../figures/'
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

# Reset plot
plt.close()

# Plot mean and standard deviation for each metric
fig, axes = plt.subplots(len(regions), len(metrics), figsize=(18, 18), sharey=False)

for j, region in enumerate(regions):
    region_data = filtered_data[filtered_data['Genesis Region'] == region]
    for i, metric in enumerate(metrics):
        ax = axes[j, i]
        
        for phase in PHASE_ORDER:
            if phase in COLOR_PHASES.keys():
                phase_data = region_data[region_data['phase'] == phase]
                mean_val = phase_data[metric].mean()
                std_val = phase_data[metric].std()
                
                ax.bar(phase, mean_val, yerr=std_val, color=COLOR_PHASES[phase], capsize=5, label=phase)
        
        if j == 0:
            ax.set_title(LATEX_LABELS[metric], fontsize=14)
        if i == 0:
            ax.set_ylabel(region, fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='both', labelsize=12)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylim(METRIC_LIMITS[metric])
        
# Add legend outside the plot
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='center left', ncol=1, bbox_to_anchor=(1, 0.5), prop={'size': 12})

plt.tight_layout()
plt.subplots_adjust(right=0.8)
plt.savefig(os.path.join(output_directory, 'average_metrics_by_region.png'))
plt.close()
