# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    summary_statistics_v2.py                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/03/02 17:31:28 by daniloceano       #+#    #+#              #
#    Updated: 2024/09/22 19:52:49 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import pandas as pd

PATH = '/Users/danilocoutodesouza/Documents/Programs_and_scripts/energetic_patterns_cyclones_south_atlantic'
base_path = f'{PATH}/csv_database_energy_by_periods'
output_directory = f'../results_chapter_5/summary_statistics/'

def read_life_cycles(base_path):
    """
    Reads all CSV files in the specified directory and collects DataFrame for each system.
    """
    systems_energetics = {}

    for filename in os.listdir(base_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(base_path, filename)
            system_id = filename.split('_')[0]
            try:
                df = pd.read_csv(file_path)
                df = df.rename(columns={'Unnamed: 0': 'phase'})
                df.index = range(1, len(df) + 1)
                systems_energetics[system_id] = df
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return systems_energetics

def compute_summary_statistics(systems_energetics, output_directory):
    """
    Computes summary statistics for each term in the energetic data and exports the results as a LaTeX table.
    """
    all_data = pd.concat(systems_energetics.values(), ignore_index=True)
    terms = [col for col in all_data.columns if col not in ['Unnamed: 0', 'phase', 'system_id']]

    summary_stats = []

    for term in terms:
        term_data = all_data[term].dropna() if term not in ['Az', 'Ae', 'Kz', 'Ke'] else all_data[term].dropna() / 1e5
        mean = term_data.mean()
        median = term_data.median()
        std_dev = term_data.std()
        q25 = term_data.quantile(0.25)
        q75 = term_data.quantile(0.75)
        iqr = q75 - q25
        range_ = term_data.max() - term_data.min()

        # Round to 2 decimal places
        mean = round(mean, 2)
        median = round(median, 2)
        std_dev = round(std_dev, 2)
        q25 = round(q25, 2)
        q75 = round(q75, 2)
        iqr = round(iqr, 2)
        range_ = round(range_, 2)

        summary_stats.append({
            'Term': term,
            'Mean': mean,
            'Median': median,
            'Std Dev': std_dev,
            'Q20': q25,
            'Q80': q75,
            'IQR': iqr,
            'Range': range_
        })

    summary_stats_df = pd.DataFrame(summary_stats)

    # Format terms for LaTeX
    summary_stats_df['Term'] = summary_stats_df['Term'].str.replace('Phi', r'$\Phi$', regex=False)
    summary_stats_df['Term'] = summary_stats_df['Term'].str.replace(r' \(finite diff\.\)', '', regex=True)
    summary_stats_df['Term'] = summary_stats_df['Term'].str.replace('/', r'/', regex=False)

    # Export to LaTeX
    latex_table = summary_stats_df.to_latex(index=False, escape=False, float_format="%.2f")
    os.makedirs(output_directory, exist_ok=True)
    with open(os.path.join(output_directory, 'summary_statistics.tex'), 'w') as f:
        f.write(latex_table)

    print(f"Summary statistics saved to {os.path.join(output_directory, 'summary_statistics_total_v2.tex')}")

if __name__ == "__main__":
    systems_energetics = read_life_cycles(base_path)
    compute_summary_statistics(systems_energetics, output_directory)