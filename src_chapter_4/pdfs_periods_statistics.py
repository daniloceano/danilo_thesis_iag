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

METRICS_THERSHOLDS = {
    'Total Time (h)': 0.5,
    'Mean Speed (m/s)': 0.6,
    'Total Distance (km)': 0.5,
    'Mean Vorticity (−1 × 10−5 s−1)': 0.9,
    'Mean Growth Rate (10^−5 s^−1 day^-1)': 0.5  
}

# Desired phase order
PHASE_ORDER = ['incipient', 'intensification', 'mature', 'decay', 'intensification 2', 'mature 2', 'decay 2']

LABELS = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)"]

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
global_xlims = {}
for metric in metrics:
    metric_data = filtered_data[metric]
    if metric != 'Mean Growth Rate (10^−5 s^−1 day^-1)':
        metric_data = metric_data[filtered_data['phase'] != 'Total']
    min_val = metric_data.min()
    max_val = metric_data.max()
    if metric != 'Mean Growth Rate (10^−5 s^−1 day^-1)':
        min_val = 0
    global_xlims[metric] = (min_val, max_val)

# Determine global y-axis limits for each metric
global_ylims = {}
for metric in metrics:
    max_y = 0
    for season in seasons:
        for region in regions:
            region_data = filtered_data[(filtered_data['Genesis Season'] == season) & 
                                        (filtered_data['Genesis Region'] == region) & 
                                        (filtered_data['phase'] != 'Total')]
            if not region_data.empty:
                y = sns.kdeplot(region_data[metric]).get_lines()[-1].get_ydata()
                kde = (y / np.sum(y) * 100)
                max_y = max(max_y, kde.max())
    global_ylims[metric] = max_y

# Reset plot
plt.close()

# Plot PDFs for each metric excluding 'Total'
for metric in metrics:
    fig, axes = plt.subplots(len(regions), len(seasons), figsize=(18, 12), sharey=False)
    
    label_mapping = {
        'incipient': 'Ic',
        'incipient 2': 'Ic2',
        'intensification': 'It',
        'intensification 2': 'It2',
        'mature': 'M',
        'mature 2': 'M2',
        'decay': 'D',
        'decay 2': 'D2',
    }

    for i, season in enumerate(seasons):
        for j, region in enumerate(regions):
            ax = axes[j, i]
            region_data = filtered_data[(filtered_data['Genesis Season'] == season) & 
                                        (filtered_data['Genesis Region'] == region) & 
                                        (filtered_data['phase'] != 'Total')]
            
            sorted_phases = sorted(region_data['phase'].unique(), key=lambda x: PHASE_ORDER.index(x) if x in PHASE_ORDER else len(PHASE_ORDER))
            
            for phase in sorted_phases:
                phase_data = region_data[region_data['phase'] == phase]
                line_style = '--' if '2' in phase else '-'
                kdeplot = sns.kdeplot(phase_data[metric], ax=ax, label=phase, color=COLOR_PHASES.get(phase), linestyle=line_style, linewidth=3)
                
                # Normalize the y-axis to show percentages
                y = kdeplot.get_lines()[-1].get_ydata()
                kdeplot.get_lines()[-1].set_ydata(y / np.sum(y) * 100)
                
                # Calculate mean and standard deviation
                mean_val = phase_data[metric].mean()
                std_val = phase_data[metric].std()
                ax.text(0.95, 0.92 - (0.1 * sorted_phases.index(phase)), 
                        f'{label_mapping.get(phase, phase)}: {mean_val:.2f} ± {std_val:.2f}', 
                        color=COLOR_PHASES.get(phase), transform=ax.transAxes, ha='right', fontsize=12, fontweight='bold')

            # Calculate label index
            label_index = j * len(seasons) + i
            ax.text(0.01, 1.05, f'{LABELS[label_index]}', transform=ax.transAxes, fontsize=14, fontweight='bold')

            ax.set_title(f'{season} - {region}', fontsize=14, fontweight='bold')
            ax.set_xlabel(LATEX_LABELS[metric], fontsize=14)
            ax.set_ylabel('Percentage (%)', fontsize=14)
            ax.set_xlim(lim * METRICS_THERSHOLDS[metric] for lim in global_xlims[metric])
            ax.set_ylim(0, global_ylims[metric] + 0.1)
            ax.tick_params(axis='both', labelsize=12)

            # Add gridlines
            ax.grid(axis='both', linestyle='--', linewidth=1, alpha=0.5)

            # Add vertical line at 0 for 'Mean Growth Rate'
            if metric == 'Mean Growth Rate (10^−5 s^−1 day^-1)':
                for ax in axes[:, i]:
                    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Move legend to the right outside the plot
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', ncol=7, bbox_to_anchor=(0.08, 0.025), prop={'size': 14})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    metric_name = METRIC_NAMES[metric]
    plt.savefig(os.path.join(output_directory, f'pdf_{metric_name}.png'))
    plt.close()
