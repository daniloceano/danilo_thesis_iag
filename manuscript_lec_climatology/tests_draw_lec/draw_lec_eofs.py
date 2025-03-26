# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    draw_lec_eofs.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/01/03 23:31:13 by daniloceano       #+#    #+#              #
#    Updated: 2025/03/26 07:47:41 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from draw_lec_v6 import plot_period_means

def load_eofs_data(base_path, results_path):
    """
    Carrega todos os dados de todas as EOFs para todas as fases.
    Retorna um dicionário de DataFrames, com cada fase contendo um DataFrame com os dados da fase.
    """

    results = glob(results_path+'/*.csv')
    dummy_result = pd.read_csv(results[0], index_col=0)
    columns = dummy_result.columns

    # Definir as fases
    phases = ['total', 'decay', 'decay 2', 'incipient', 'intensification', 'intensification 2', 'mature', 'mature 2']

    eofs_data = {}

    # Percorrer cada fase e carregar os dados
    for phase in phases:
        phase_path = os.path.join(base_path, phase)
        
        # Certificar-se de que a pasta existe
        if os.path.exists(phase_path):
            # Carregar os arquivos relevantes para cada fase
            eofs_file = os.path.join(phase_path, 'eofs.csv')
            df_eof = pd.read_csv(eofs_file)
            df_eof.columns = columns
            df_eof.index_name = 'EOF'
            df_eof.index = df_eof.index + 1

            # Carregar os arquivos em DataFrames
            eofs_data[phase] = df_eof

        else:
            print(f"Warning: The folder for phase {phase} does not exist!")

    return eofs_data

def create_individual_eof_dataframes(eofs_data):
    """
    Cria DataFrames separados para cada EOF onde as fases são os índices e os termos são as colunas.
    """

    dsf_eof = {}
    phases = eofs_data.keys()
    eofs = eofs_data['total'].index

    # Inicializar um df para cada EOF
    for eof in eofs:
        dsf_eof[eof] = pd.DataFrame(index=phases, columns=eofs_data['total'].columns)
        
    for phase in phases:
        df_phase = eofs_data[phase]
        for eof in eofs:
            dsf_eof[eof].loc[phase] = df_phase.loc[eof]

    return dsf_eof

def plot_lorenzcycletoolkit(periods_df, figures_directory, eof):

    # Rename columns by removing "(finite diff.)"
    periods_df = periods_df.rename(columns=lambda x: x.replace(" (finite diff.)", ""))
    plot_period_means(periods_df, "min_max")

    # Anotar número da EOF no centro da figura
    plt.text(0.51, 0.9, f"EOF {eof}", transform=plt.gcf().transFigure, fontsize=20, ha='center', va='center', fontweight='bold')

    figures_subdirectory = os.path.join(figures_directory, "draw_LEC")
    os.makedirs(figures_subdirectory, exist_ok=True)
    figure_path = os.path.join(figures_subdirectory, f"LEC_eof_{eof}.png")
    plt.savefig(figure_path)
    plt.close()
    print(f"Lorenz cycle plot saved to {figure_path}")


if __name__ == "__main__":
    # Test for Reg1-Representative_fixed
    PATH = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic'
    results_path = f'{PATH}/csv_database_energy_by_periods'
    eofs_path = f'{PATH}/csv_eofs_energetics_with_track'
    figures_directory = "./figures/"

    groups = {
        'Energy Terms': ['A', 'K'],
        'Conversion Terms': ['C'],
        'Boundary Terms': ['BA', 'BK'],
        'Pressure Work Terms': ['BΦ'],
        'Generation/Residual Terms': ['G', 'R'],
        'Budget Terms': ['∂']
    }

    terms_prefix = list(groups.keys())

    eofs_data = load_eofs_data(eofs_path, results_path)

    dfs_by_eof = create_individual_eof_dataframes(eofs_data)

    # Plot Lorenz cycle
    for eof in dfs_by_eof.keys():
        periods_df = dfs_by_eof[eof]
        # Transformar todas as colunas para tipo float
        periods_df = periods_df.apply(pd.to_numeric, errors='coerce')
        plot_lorenzcycletoolkit(periods_df, figures_directory, eof)