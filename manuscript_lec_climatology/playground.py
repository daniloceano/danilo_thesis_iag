"""
Analyze principal components (PCs) and energetics for intense cyclones
in the South Atlantic, classified by dominant EOFs and vorticity percentiles.

Author: Danilo Couto de Souza
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_merge_pcs(pcs_q10_path, pcs_q90_path):
    """Load PCs with dominant EOFs and merge q10 and q90 data."""
    pcs_q10 = pd.read_csv(pcs_q10_path)
    pcs_q90 = pd.read_csv(pcs_q90_path)

    pcs_q10.rename(columns={'dominant_eof': 'dominant_eof_q10'}, inplace=True)
    pcs_q90.rename(columns={'dominant_eof': 'dominant_eof_q90'}, inplace=True)

    merged_df = pd.merge(pcs_q10, pcs_q90, on='track_id', how='outer', suffixes=('_q10', '_q90'))
    merged_df['dominant_eof_q10'].fillna(0, inplace=True)
    merged_df['dominant_eof_q90'].fillna(0, inplace=True)

    for col in [f'PC{i}' for i in range(1, 9)]:
        merged_df[f'{col}_q10'].fillna(merged_df[f'{col}_q90'], inplace=True)
        merged_df.drop(columns=[f'{col}_q90'], inplace=True)
        merged_df.rename(columns={f'{col}_q10': col}, inplace=True)

    return merged_df


def classify_intensity(df, variable='vor42', percentiles=[0, 15, 35, 65, 85, 100]):
    """Classify systems by intensity based on percentile bins."""
    intensity_labels = [f'{percentiles[i]}-{percentiles[i+1]}' for i in range(len(percentiles)-1)]
    percentile_values = np.percentile(df[variable].dropna(), percentiles[1:-1])
    df['intensity_category'] = pd.cut(df[variable], 
                                      bins=[-np.inf] + list(percentile_values) + [np.inf],
                                      labels=intensity_labels)
    return df


def add_time_columns(df):
    """Extract year and 5-year bins from track_id."""
    df['year_from_track'] = df['track_id'].astype(str).str[:4].astype(int)
    bins = list(range(1980, 2025, 5))
    labels = [f'{start}-{start+4}' for start in bins[:-1]]
    df['five_years'] = pd.cut(df['year_from_track'], bins=bins, labels=labels, right=False)
    return df


def plot_pc_series(df):
    """Plot time series and boxplots of PCs for intense cyclones."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    for pc in ['PC1', 'PC2', 'PC3', 'PC4']:
        axes[0].plot(df['track_id'], df[pc], label=pc)
    axes[0].set_title('PC1 to PC4 Time Series')
    axes[0].set_xlabel('Track ID')
    axes[0].set_ylabel('PC Value')
    axes[0].legend()

    axes[1].boxplot([df[pc] for pc in ['PC1', 'PC2', 'PC3', 'PC4']])
    axes[1].set_title('PC1 to PC4 Boxplots')
    axes[1].set_xticklabels(['PC1', 'PC2', 'PC3', 'PC4'])
    axes[1].set_ylabel('PC Value')

    plt.tight_layout()
    plt.show()


def plot_stacked_pcs(df):
    """Plot stacked bar chart of PC1 to PC4 values for intense systems with custom colors."""
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Dados e cores customizadas
    stacked_data = df[['PC1', 'PC2', 'PC3', 'PC4']]
    custom_colors = {
        'PC1': '#5975A4',  # Azul
        'PC2': '#CC8963',  # Amarelo queimado
        'PC3': '#B55D60',  # Vermelho
        'PC4': '#5F9E6E'   # Verde
    }

    # Plot com cores definidas
    stacked_data.plot(kind='bar', stacked=True, ax=ax, width=1, color=[custom_colors[col] for col in stacked_data.columns])

    ax.set_title('PC1 to PC4 Time Series – Cyclones with ζ_central > q99')
    ax.set_xlabel('Cyclones')
    ax.set_ylabel('PC Value')
    # ax.set_xticks([])
    ax.set_xticklabels([])
    ax.legend(title='Principal Components')
    plt.tight_layout()
    plt.savefig('figures/eof_statistics/intense_systems_PCs_stacked_pcs.png', dpi=300)


def plot_intensity_distribution(df):
    """Plot bar charts of cyclone intensity distribution by 5-year bins."""
    intensity_counts = pd.crosstab(df['five_years'], df['intensity_category'])
    intensity_counts.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='tab10')
    plt.title('Cyclone Intensity per 5-Year Period')
    plt.xlabel('5-Year Bin')
    plt.ylabel('Number of Cyclones')
    plt.tight_layout()
    plt.show()


def plot_individual_intensity_panels(df, intensity_labels):
    """Plot one bar chart per intensity category."""
    fig, axes = plt.subplots(1, len(intensity_labels), figsize=(20, 6), sharey=True)

    for i, category in enumerate(intensity_labels):
        category_data = df[df['intensity_category'] == category]
        category_counts = category_data['five_years'].value_counts().sort_index()
        axes[i].bar(category_counts.index, category_counts.values, color='lightblue')
        axes[i].set_title(f'Intensity: {category}')
        axes[i].set_xlabel('5-Year Bin')
        axes[i].tick_params(axis='x', rotation=45)
        if i == 0:
            axes[i].set_ylabel('Number of Cyclones')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # === File paths ===
    PATH = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic'
    pcs_q10_path = f'{PATH}/csv_eofs_energetics_with_track/Total/pcs_with_dominant_eof_q10.csv'
    pcs_q90_path = f'{PATH}/csv_eofs_energetics_with_track/Total/pcs_with_dominant_eof_q90.csv'
    energetics_path = f'{PATH}/tracks_SAt_filtered/tracks_SAt_filtered_with_energetics.csv'

    # === Load and process data ===
    merged_df = load_and_merge_pcs(pcs_q10_path, pcs_q90_path)
    energetics_df = pd.read_csv(energetics_path)

    # === Filter intense systems and merge PCs ===
    max_vor_per_track = energetics_df.groupby('track_id')['vor42'].max()
    vor_threshold = max_vor_per_track.quantile(0.95)
    intense_ids = max_vor_per_track[max_vor_per_track > vor_threshold].index
    intense_merged_df = merged_df[merged_df['track_id'].isin(intense_ids)]

    # === Visualizations ===
    # plot_pc_series(intense_merged_df)
    plot_stacked_pcs(intense_merged_df)

    # === Intensity classification and temporal analysis ===
    energetics_df = classify_intensity(energetics_df)
    energetics_df = add_time_columns(energetics_df)

    # plot_intensity_distribution(energetics_df)
    # plot_individual_intensity_panels(energetics_df, ['0-15', '15-35', '35-65', '65-85', '85-100'])
