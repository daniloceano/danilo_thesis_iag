import os
import pandas as pd
import numpy as np
from glob import glob
from plot_LEC_std import plot_lorenzcycletoolkit_with_std 

PATH = "/home/daniloceano/Documents/Programs_and_scripts/energetic_patterns_cyclones_south_atlantic/"
results_path = f'{PATH}/csv_database_energy_by_periods'
eofs_path = f'{PATH}/csv_eofs_energetics'
output_directory = '../figures/LEC_eof/'

phases = ['incipient', 'intensification', 'mature', 'decay', 'intensification 2', 'mature 2', 'decay 2', 'Total']

# Função para normalizar os dados
def normalize_idata_not_energy(data):
    df_not_energy = np.abs(data.drop(columns=['Az', 'Ae', 'Kz', 'Ke']))
    normalized_data_not_energy = ((df_not_energy - df_not_energy.min()) / (df_not_energy.max() - df_not_energy.min()))
    normalized_data_not_energy = df_not_energy.clip(lower=1.5, upper=15)
    return normalized_data_not_energy

# DataFrame vazio para armazenar todos os resultados
all_eofs_data = pd.DataFrame()

for phase in phases:

    # Leia o primeiro arquivo CSV para obter as colunas
    results = glob(results_path+'/*.csv')
    dummy_result = pd.read_csv(results[0], index_col=0)
    columns = dummy_result.columns

    # Carregue os dados de EOF para a fase
    phase_directory = os.path.join(eofs_path, phase)
    eof_file_phase = os.path.join(phase_directory, f'eofs.csv')
    df = pd.read_csv(eof_file_phase, header=None)

    # Ajuste os nomes das colunas
    df.columns = columns
    df.columns = [col if '∂' not in col else '∂' + col.split('∂')[1].split('/')[0] + '/∂t' for col in df.columns]

    # Normalização dos dados
    normalized_data_not_energy = normalize_idata_not_energy(df)

    # Carregar a variância explicada
    explained_variance = pd.read_csv(os.path.join(phase_directory, 'variance_fraction.csv'), header=None)

    # Processar cada EOF e adicionar ao DataFrame principal
    for eof in range(len(df)):

        # Obtenha os dados para o EOF atual
        idata = df.iloc[eof]

        # Adicionar o número da fase e a variância explicada
        explained_variance_eof_percentage = round(float(explained_variance.iloc[eof].values[0] * 100), 2)
        idata.name = f'EOF {eof+1} - {phase.capitalize()} - Exp. Var.: {explained_variance_eof_percentage}%'

        # Criação de um DataFrame para o EOF da fase atual
        eof_data = idata.to_frame().T  # Transforma em DataFrame de uma linha
        eof_data['Phase'] = phase  # Adiciona a coluna da fase

        # Anexar os dados do EOF no DataFrame principal
        all_eofs_data = pd.concat([all_eofs_data, eof_data])


# Dicionário para armazenar os DataFrames de cada EOF com todas as fases
eof_dataframes = {}

# Iterar sobre os índices de all_eofs_data para agrupar os dados por cada EOF
for eof in range(1, 4):  # Para cada EOF, onde o índice é único
    # Filtrar os dados do all_eofs_data para o EOF atual
    eof_data = all_eofs_data[all_eofs_data.index.str.contains(f"EOF {eof}")]

    # Adicionar os dados filtrados ao dicionário com o nome do EOF
    eof_dataframes[eof] = eof_data

# Exemplo de como acessar o DataFrame para um EOF específico com todas as fases
eof_1_all_phases_df = eof_dataframes.get('EOF 1')