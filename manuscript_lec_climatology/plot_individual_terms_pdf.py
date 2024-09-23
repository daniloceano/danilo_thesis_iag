import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

PATH = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic'
base_path = f'{PATH}/csv_database_energy_by_periods'
output_directory = f'./figures/individual_pdfs/'

COLOR_TERMS = ["#3B95BF", "#87BF4B", "#BFAB37", "#BF3D3B", "#873e23", "#A13BF0"]

# Global variables for font sizes
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 12
TITLE_FONT_SIZE = 16
LEGEND_FONT_SIZE = 12

def read_life_cycles(base_path):
    """
    Reads all CSV files in the specified directory and collects DataFrame for each system.
    """
    systems_energetics = {}

    for filename in tqdm(os.listdir(base_path), desc="Reading CSV files"):
        if filename.endswith('.csv'):
            file_path = os.path.join(base_path, filename)
            system_id = filename.split('_')[0]
            try:
                df = pd.read_csv(file_path)
                systems_energetics[system_id] = df
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return systems_energetics

def plot_individual_pdfs(systems_energetics, output_directory):
    """
    Plots the PDF for each term in the dataset in individual figures.

    Parameters:
    - systems_energetics: Dictionary of DataFrames with system data.
    - output_directory: Directory to save the plots.
    """
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Get the terms for the group
    terms = systems_energetics[list(systems_energetics.keys())[0]].drop(columns=['Unnamed: 0']).columns.to_list()

    # Concatenate all systems' dataframes while retaining the system id and phase
    all_data = pd.concat([df.assign(system_id=system_id) for system_id, df in systems_energetics.items()])
    all_data.rename(columns={'Unnamed: 0': 'Phase'}, inplace=True)

    # Convert relevant columns to numeric, forcing errors to NaN
    all_data = all_data.apply(pd.to_numeric, errors='coerce')

    # Compute mean across all phases for each system
    mean_data = all_data.drop('Phase', axis=1).groupby('system_id').mean().reset_index()

    # Melt the dataframe for plotting
    mean_data_melted = mean_data.melt(id_vars=['system_id'], var_name='Term', value_name='Value')

    # Plot the PDF for each term
    for idx, term in enumerate(terms):
        term_data = mean_data_melted[mean_data_melted['Term'] == term]
        if not term_data.empty:
            plt.figure(figsize=(8, 6))
            sns.kdeplot(data=term_data, x='Value', bw_adjust=0.5, fill=True, color=COLOR_TERMS[idx % len(COLOR_TERMS)])
            plt.title(f'PDF for {term}', fontsize=TITLE_FONT_SIZE)
            plt.xlabel('Value', fontsize=LABEL_FONT_SIZE)
            plt.ylabel('Density', fontsize=LABEL_FONT_SIZE)
            plt.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
            
            # Save the plot
            plot_filename = f'{term}_pdf.png'.replace('/', '_')
            plot_path = os.path.join(output_directory, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved {plot_filename} in {output_directory}")

if __name__ == "__main__":
    systems_energetics = read_life_cycles(base_path)
    plot_individual_pdfs(systems_energetics, output_directory)
