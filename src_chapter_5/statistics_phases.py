# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    statistics_phases.py                               :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/22 15:06:46 by daniloceano       #+#    #+#              #
#    Updated: 2024/06/22 13:22:26 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import pandas as pd
import numpy as np
from scipy.stats import f_oneway, kruskal, shapiro, levene
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
from tqdm import tqdm

PATH = '/Users/danilocoutodesouza/Documents/Programs_and_scripts/energetic_patterns_cyclones_south_atlantic'
base_path = f'{PATH}/csv_database_energy_by_periods'
output_directory = f'../results_chapter_5/statistics_phases/'

PHASE_ORDER = ['Total', 'incipient', 'intensification', 'mature', 'decay', 'intensification 2', 'mature 2', 'decay 2']

groups = {
    'Energy Terms': ['A', 'K'],
    'Conversion Terms': ['C'],
    'Boundary Terms': ['B'],
    'Generation/Dissipation Terms': ['G', 'R'],
    'Budgets': ['∂']
}

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
                df = df.rename(columns={'Unnamed: 0': 'phase'})
                df.index = range(1, len(df) + 1)
                systems_energetics[system_id] = df
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return systems_energetics

def compute_total_phase(systems_energetics):
    """
    Computes the mean values across all phases for each system to represent the "Total" phase.
    """
    for system_id, df in systems_energetics.items():
        mean_values = df.mean(numeric_only=True)
        mean_values['phase'] = 'Total'
        systems_energetics[system_id] = pd.concat([df, pd.DataFrame([mean_values])], ignore_index=True)
    return systems_energetics

def check_normality(data, term):
    """
    Checks for normality using the Shapiro-Wilk test.
    """
    normality_results = {}
    for phase in PHASE_ORDER:
        stat, p_value = shapiro(data[data['phase'] == phase][term].dropna())
        normality_results[phase] = (stat, p_value)
    return normality_results

def check_homogeneity_of_variances(data, term):
    """
    Checks for homogeneity of variances using Levene's test.
    """
    phase_data = [data[data['phase'] == phase][term].dropna() for phase in PHASE_ORDER]
    stat, p_value = levene(*phase_data)
    return stat, p_value

def perform_anova_test(data, term):
    """
    Performs ANOVA test for the given term across all phases.
    """
    phase_data = [data[data['phase'] == phase][term].dropna() for phase in PHASE_ORDER]
    f_statistic, p_value = f_oneway(*phase_data)
    return f_statistic, p_value

def perform_kruskal_test(data, term):
    """
    Performs Kruskal-Wallis H test for the given term across all phases.
    """
    phase_data = [data[data['phase'] == phase][term].dropna() for phase in PHASE_ORDER]
    h_statistic, p_value = kruskal(*phase_data)
    return h_statistic, p_value

def perform_post_hoc_tests(data, term, test_type):
    """
    Performs post-hoc tests if ANOVA/Kruskal-Wallis test is significant.
    """
    post_hoc_results = {}
    all_data = data[['phase', term]].dropna()
    
    if test_type == 'ANOVA':
        # Tukey HSD for ANOVA
        tukey = pairwise_tukeyhsd(endog=all_data[term], groups=all_data['phase'], alpha=0.05)
        post_hoc_results['tukey'] = tukey.summary()
    else:
        # Dunn's Test for Kruskal-Wallis
        phase_data = [data[data['phase'] == phase][term].dropna() for phase in PHASE_ORDER]
        dunn = sp.posthoc_dunn(phase_data, p_adjust='bonferroni')
        post_hoc_results['dunn'] = dunn

    return post_hoc_results

def generate_latex_table(results):
    """
    Generates a LaTeX table from the statistical results.
    """
    latex_table = """
\\begin{table}[!htbp]
\\centering
\\label{tab:lec_stats}
\\begin{tabular}{lrrrrr}
\\toprule
\\textbf{Term} & \\textbf{KW Statistic} & \\textbf{KW p-value} & \\textbf{Normality p-value} & \\textbf{Homogeneity Statistic} & \\textbf{Homogeneity p-value} \\\\
\\midrule
"""
    
    for result in results:
        term = result['Term']
        kw_stat = result['Kruskal-Wallis Statistic']
        kw_p_value = result['Kruskal-Wallis p-value']
        normality_p_value = result['Normality Results']['Total'][1]
        homogeneity_stat = result['Homogeneity Statistic']
        homogeneity_p_value = result['Homogeneity p-value']

        if '∂' in term:
            tmp = term[1:3]
            term = fr'$\frac{{\partial {tmp}}}{{\partial t}}$'
        term = term.replace('BΦZ', r'$B\Phi Z$')
        term = term.replace('BΦE', r'$B\Phi E$')
        
        latex_table += f"{term} & {kw_stat:.2f} & {kw_p_value:.2e} & {normality_p_value:.2e} & {homogeneity_stat:.2f} & {homogeneity_p_value:.2e} \\\\\n"
    
    latex_table += """
\\bottomrule
\\bottomrule
\\end{tabular}
\\caption{Statistical Analysis Results for Different Phases. The table shows the Kruskal-Wallis (KW) test statistic and p-value for each term, along with the p-value for the Shapiro-Wilk normality test on the Total phase and the Levene's test statistic and p-value for homogeneity of variances. The Kruskal-Wallis test is used here due to non-normality or heterogeneity of variances in the data, indicating significant differences among phases if the p-value is less than 0.05.}
\\label{tab:stat_analysis}
\\end{table}
"""
    
    return latex_table

def analyze_statistics(systems_energetics, terms, output_directory):
    """
    Analyzes the statistics for the specified terms and saves the results to CSV files.
    """
    all_data = pd.concat(systems_energetics.values(), ignore_index=True)
    results = []

    for term in terms:
        print(f"Analyzing {term}...")

        # Check assumptions
        normality_results = check_normality(all_data, term)
        homogeneity_stat, homogeneity_p_value = check_homogeneity_of_variances(all_data, term)

        # Determine test to use based on assumptions
        if all(p_value > 0.05 for _, p_value in normality_results.values()) and homogeneity_p_value > 0.05:
            test_type = 'ANOVA'
            f_statistic, p_value = perform_anova_test(all_data, term)
        else:
            test_type = 'Kruskal-Wallis'
            f_statistic, p_value = perform_kruskal_test(all_data, term)

        result = {
            'Term': term,
            'Kruskal-Wallis Statistic': f_statistic,
            'Kruskal-Wallis p-value': p_value,
            'Normality Results': normality_results,
            'Homogeneity Statistic': homogeneity_stat,
            'Homogeneity p-value': homogeneity_p_value
        }

        if p_value < 0.05:
            post_hoc_results = perform_post_hoc_tests(all_data, term, test_type)
            result['Post-hoc Results'] = post_hoc_results

        results.append(result)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    os.makedirs(output_directory, exist_ok=True)
    results_df.to_csv(os.path.join(output_directory, 'statistical_analysis_results.csv'), index=False)

    # Generate LaTeX table
    latex_table = generate_latex_table(results)
    with open(os.path.join(output_directory, 'statistical_analysis_results.tex'), 'w') as f:
        f.write(latex_table)

    print(f"Saved statistical analysis results in {output_directory}")

def analyze_all_groups(systems_energetics, groups, output_directory):
    """
    Analyzes the statistics for all groups and their respective terms.
    """
    all_data = pd.concat(systems_energetics.values(), ignore_index=True)

    # Preprocess budget terms to remove "(finite diff.)"
    all_data.columns = [col.replace(" (finite diff.)", "") for col in all_data.columns]

    results = []

    for group_name, terms_prefix in groups.items():
        print(f"Analyzing group: {group_name}...")
        
        terms = [col for col in all_data.columns if any(col.startswith(prefix) for prefix in terms_prefix)]
        
        for term in terms:
            print(f"  Analyzing {term}...")

            # Check assumptions
            normality_results = check_normality(all_data, term)
            homogeneity_stat, homogeneity_p_value = check_homogeneity_of_variances(all_data, term)

            # Determine test to use based on assumptions
            if all(p_value > 0.05 for _, p_value in normality_results.values()) and homogeneity_p_value > 0.05:
                test_type = 'ANOVA'
                f_statistic, p_value = perform_anova_test(all_data, term)
            else:
                test_type = 'Kruskal-Wallis'
                f_statistic, p_value = perform_kruskal_test(all_data, term)

            result = {
                'Term': term,
                'Group': group_name,
                'Kruskal-Wallis Statistic': f_statistic,
                'Kruskal-Wallis p-value': p_value,
                'Normality Results': normality_results,
                'Homogeneity Statistic': homogeneity_stat,
                'Homogeneity p-value': homogeneity_p_value
            }

            if p_value < 0.05:
                post_hoc_results = perform_post_hoc_tests(all_data, term, test_type)
                result['Post-hoc Results'] = post_hoc_results

            results.append(result)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    os.makedirs(output_directory, exist_ok=True)
    results_df.to_csv(os.path.join(output_directory, 'statistical_analysis_results.csv'), index=False)

    # Generate LaTeX table
    latex_table = generate_latex_table(results)
    with open(os.path.join(output_directory, 'statistical_analysis_results.tex'), 'w') as f:
        f.write(latex_table)

    print(f"Saved statistical analysis results in {output_directory}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    os.makedirs(output_directory, exist_ok=True)
    results_df.to_csv(os.path.join(output_directory, 'statistical_analysis_results.csv'), index=False)

    # Generate LaTeX table
    latex_table = generate_latex_table(results)
    with open(os.path.join(output_directory, 'statistical_analysis_results.tex'), 'w') as f:
        f.write(latex_table)

    print(f"Saved statistical analysis results in {output_directory}")

if __name__ == "__main__":
    os.makedirs(output_directory, exist_ok=True)

    systems_energetics = read_life_cycles(base_path)
    systems_energetics = compute_total_phase(systems_energetics)

    analyze_all_groups(systems_energetics, groups, output_directory)

