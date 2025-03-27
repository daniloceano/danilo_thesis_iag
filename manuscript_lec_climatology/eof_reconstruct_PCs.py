import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Função para filtrar os track_ids com base no limiar da magnitude das PCs
def get_track_ids_with_eof(df, eof_column, threshold=2):
    """
    Filtra os sistemas (track_id) associados ao EOF dominante, com base na magnitude das PCs.
    
    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os valores das PCs e a coluna de EOF dominante.
    eof_column (str): Nome da coluna que contém os valores de EOF dominante ('dominant_eof_q10' ou 'dominant_eof_q90').
    threshold (float): O limiar de multiplicação que define a PC dominante como pelo menos 'threshold' vezes maior 
                       que as demais PCs. O padrão é 5.
    
    Retorna:
    pd.DataFrame: Um DataFrame com track_ids e os valores das PCs associadas ao EOF dominante, para aqueles
                  que atendem ao critério de magnitude.
    """
    track_info = []

    # Loop para cada EOF (1, 2, 3, 4)
    for eof in range(1, 5):
        # Filtrar os dados para o EOF dominante
        filtered_df = df[df[eof_column] == eof]
        
        # Comparar as magnitudes das PCs, selecionando aquelas que são pelo menos 'threshold' vezes maiores que as demais
        for track_id in filtered_df['track_id'].unique():
            track_data = filtered_df[filtered_df['track_id'] == track_id]
            pcs_values = track_data[['PC1', 'PC2', 'PC3', 'PC4']].values.flatten()
            
            # Calcular as magnitudes das PCs
            pcs_magnitudes = np.abs(pcs_values)
            
            # Identificar a PC dominante (com base no valor do EOF)
            dominant_pc_index = eof - 1  # Porque os índices das PCs começam de 0
            dominant_pc_value = pcs_values[dominant_pc_index]
            
            # Verificar se a magnitude da PC dominante é pelo menos 'threshold' vezes maior que as outras
            if np.all(np.abs(dominant_pc_value) >= threshold * np.delete(pcs_magnitudes, dominant_pc_index)):
                track_info.append({
                    'track_id': track_id,
                    'dominant_eof': eof,
                    'PC1': pcs_values[0],
                    'PC2': pcs_values[1],
                    'PC3': pcs_values[2],
                    'PC4': pcs_values[3]
                })
    
    # Converter a lista para um DataFrame
    track_df = pd.DataFrame(track_info)
    return track_df

# Função para testar diferentes limiares e plotar os resultados
def test_thresholds_and_plot(df, thresholds):
    """
    Testa diferentes limiares e plota a contagem de sistemas detectados para cada limiar.
    
    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os valores das PCs e a coluna de EOF dominante.
    thresholds (list): Lista de valores de limiares (thresholds) a serem testados.
    
    Retorna:
    None: Gera e exibe um gráfico de barras com a contagem de sistemas detectados para cada limiar.
    """
    # Armazenar os resultados para cada limiar
    results_q10 = []
    results_q90 = []

    # Para cada limiar, obter a contagem de sistemas detectados para q10 e q90
    for threshold in thresholds:
        # Para q10
        track_df_q10 = get_track_ids_with_eof(df, 'dominant_eof_q10', threshold)
        detected_count_q10 = len(track_df_q10)
        results_q10.append({'Threshold': threshold, 'Detected Systems': detected_count_q10})
        
        # Para q90
        track_df_q90 = get_track_ids_with_eof(df, 'dominant_eof_q90', threshold)
        detected_count_q90 = len(track_df_q90)
        results_q90.append({'Threshold': threshold, 'Detected Systems': detected_count_q90})
    
    # Converter os resultados em DataFrames
    results_df_q10 = pd.DataFrame(results_q10)
    results_df_q90 = pd.DataFrame(results_q90)

    # Plotar os resultados com barras lado a lado
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotando as barras para q10 e q90 lado a lado
    bar_width = 0.35
    index = np.arange(len(thresholds))

    # Barras para q10 e q90 com as cores especificadas
    bar1 = ax.bar(index, results_df_q10['Detected Systems'], bar_width, label='EOF_q10', color='#5975A4')
    bar2 = ax.bar(index + bar_width, results_df_q90['Detected Systems'], bar_width, label='EOF_q90', color='#B55D60')

    # Adicionando detalhes ao gráfico
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Detected Systems')
    ax.set_title('Detected Systems for Each Threshold (q10 and q90)')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(thresholds)
    ax.legend()

    # Exibir o gráfico
    plt.tight_layout()
    plt.show()

def get_top_3_systems_from_filtered_df(df, threshold_column):
    """
    Pega os 3 sistemas com o maior sinal de cada PC (PC1, PC2, PC3, PC4) a partir de um DataFrame filtrado,
    como track_df_threshold_5_q10 ou track_df_threshold_5_q90.
    
    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os valores das PCs e a coluna de EOF dominante.
    threshold_column (str): Nome da coluna de EOF dominante filtrado ('dominant_eof_q10' ou 'dominant_eof_q90').
    
    Retorna:
    pd.DataFrame: DataFrame com os 3 sistemas com o maior sinal de cada PC, para o limiar 5.
    """
    top_3_systems = []

    # Para cada EOF (1, 2, 3, 4)
    for eof in range(1, 5):
        # Filtrar os dados para o EOF dominante
        filtered_df = df[df[threshold_column] == eof]
        
        # Identificar as PCs correspondentes a cada EOF
        for pc in ['PC1', 'PC2', 'PC3', 'PC4']:
            # Ordenar os sistemas pela magnitude (valor absoluto) da PC em ordem decrescente
            filtered_df['PC_magnitude'] = np.abs(filtered_df[pc])  # Calcula a magnitude
            top_3 = filtered_df.nlargest(3, 'PC_magnitude')  # Pega os 3 maiores valores de magnitude
            
            # Garantir que pegamos no máximo 3 sistemas
            top_3 = top_3.head(3)  # Limita o DataFrame a no máximo 3 linhas
            
            # Adicionar ao resultado, mantendo as colunas de PCs
            for _, row in top_3.iterrows():
                top_3_systems.append({
                    'track_id': row['track_id'],
                    'dominant_eof': eof,
                    'PC1': row['PC1'],
                    'PC2': row['PC2'],
                    'PC3': row['PC3'],
                    'PC4': row['PC4']
                })
    
    # Converter a lista para um DataFrame
    top_3_df = pd.DataFrame(top_3_systems)
    return top_3_df

def reconstruct_energetics_from_pcs_and_eofs(top_3_df, eofs):
    """
    Reconstrói a energética de cada sistema usando as PCs e os coeficientes das EOFs.
    
    Parâmetros:
    top_3_df (pd.DataFrame): DataFrame com os 3 sistemas e seus valores das PCs.
    eofs (pd.DataFrame): DataFrame contendo os coeficientes das EOFs.
    
    Retorna:
    pd.DataFrame: DataFrame com a energética reconstruída para cada sistema.
    """
    reconstructed_dfs = []

    # Para cada sistema no top_3_df
    for _, row in top_3_df.iterrows():
        track_id = row['track_id']
        dominant_eof = row['dominant_eof']
        
        # Extrair os coeficientes das EOFs para o EOF dominante (linha correspondente ao EOF)
        eof_coefficients = eofs.iloc[int(dominant_eof)] 
        
        # Multiplicar as PCs pelos coeficientes das EOFs
        energetic_terms = {}
        for pc in ['PC1', 'PC2', 'PC3', 'PC4']:
            energetic_terms[pc] = row[pc] * eof_coefficients
        
        # Calcular a soma ponderada (energetic_value)
        energetic_value = sum(energetic_terms.values())
        
        # Adicionar o resultado ao DataFrame de reconstrução
        df_reconstructed = pd.DataFrame(energetic_value)
        df_reconstructed = df_reconstructed.T
        df_reconstructed['track_id'] = track_id
        df_reconstructed['dominant_eof'] = dominant_eof
        reconstructed_dfs.append(df_reconstructed)
    
    # Converter a lista para um DataFrame
    reconstructed_df = pd.concat(reconstructed_dfs)
    return reconstructed_df

# Caminhos para os arquivos
PATH = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic'
pcs_q10_path = f'{PATH}/csv_eofs_energetics_with_track/Total/pcs_with_dominant_eof_q10.csv'
pcs_q90_path = f'{PATH}/csv_eofs_energetics_with_track/Total/pcs_with_dominant_eof_q90.csv'
eofs_path = f'{PATH}/csv_eofs_energetics_with_track/Total/eofs.csv'

# Carregar os dados
pcs_q10 = pd.read_csv(pcs_q10_path)
pcs_q90 = pd.read_csv(pcs_q90_path)
eofs = pd.read_csv(eofs_path)
eofs.index = eofs.index + 1  # EOFs começam de 1

# Renomear as colunas 'dominant_eof' para 'dominant_eof_q10' e 'dominant_eof_q90'
pcs_q10.rename(columns={'dominant_eof': 'dominant_eof_q10'}, inplace=True)
pcs_q90.rename(columns={'dominant_eof': 'dominant_eof_q90'}, inplace=True)

# Realizar o merge dos dados com base no track_id, mantendo as colunas de PCs com sufixo
merged_df = pd.merge(pcs_q10, pcs_q90, on='track_id', how='outer', suffixes=('_q10', '_q90'))

# Substituir NaN por 0 nas colunas dominant_eof_q10 e dominant_eof_q90
merged_df['dominant_eof_q10'].fillna(0, inplace=True)
merged_df['dominant_eof_q90'].fillna(0, inplace=True)

# Preencher os valores de PCs ausentes com os valores de pcs_q90 onde o track_id está presente somente no pcs_q90
for col in ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8']:
    merged_df[col+'_q10'].fillna(merged_df[col+'_q90'], inplace=True)

# Remover as colunas redundantes de PCs (as do q90) após o preenchimento
for col in ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8']:
    merged_df.drop(columns=[col+'_q90'], inplace=True)

# Renomear as colunas de PCs para remover o sufixo
for col in ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8']:
    merged_df.rename(columns={col+'_q10': col}, inplace=True)

# Testar e gerar o gráfico
# thresholds = [2, 3, 4, 5]  # Testando limiares de 2 a 5
# test_thresholds_and_plot(merged_df, thresholds)

# Obter o DataFrame com as PCs usando o limiar = 5
track_df_threshold_5_q10 = get_track_ids_with_eof(merged_df, 'dominant_eof_q10', threshold=2)
track_df_threshold_5_q90 = get_track_ids_with_eof(merged_df, 'dominant_eof_q90', threshold=2)

# Exibir os DataFrames resultantes para o limiar 5
print("Track IDs com EOF_q10 e PCs para limiar 5:")
print(track_df_threshold_5_q10)

print("Track IDs com EOF_q90 e PCs para limiar 5:")
print(track_df_threshold_5_q90)

# Obter os 3 sistemas com o maior sinal de cada PC correspondente
top_3_systems_q10 = get_top_3_systems_from_filtered_df(track_df_threshold_5_q10, 'dominant_eof')
top_3_systems_q90 = get_top_3_systems_from_filtered_df(track_df_threshold_5_q90, 'dominant_eof')

# Exibir os DataFrames resultantes
print("Top 3 sistemas com maior sinal para EOF_q10:")
print(top_3_systems_q10)

print("Top 3 sistemas com maior sinal para EOF_q90:")
print(top_3_systems_q90)

# Colocar track_id como indice e int
top_3_systems_q10['track_id'] = top_3_systems_q10['track_id'].astype(int)
top_3_systems_q10 = top_3_systems_q10.set_index('track_id')

top_3_systems_q90['track_id'] = top_3_systems_q90['track_id'].astype(int)
top_3_systems_q90 = top_3_systems_q90.set_index('track_id')

# Remover duplicados
top_3_systems_q10 = top_3_systems_q10[~top_3_systems_q10.index.duplicated(keep='first')]
top_3_systems_q90 = top_3_systems_q90[~top_3_systems_q90.index.duplicated(keep='first')]

# Reordenar os índices por eof dominante e track_id
top_3_systems_q10 = top_3_systems_q10.sort_values(by=['dominant_eof', 'track_id'])
top_3_systems_q90 = top_3_systems_q90.sort_values(by=['dominant_eof', 'track_id'])

# Abrir dados das tracks com energética
energetics_path = f'{PATH}/tracks_SAt_filtered/tracks_SAt_filtered_with_energetics.csv'
energetics_df = pd.read_csv(energetics_path)

# Filtrar as tracks com base nos track_ids dos sistemas com limiar 5
energetics_q10 = energetics_df[energetics_df['track_id'].isin(top_3_systems_q10.index)].dropna()
energetics_q90 = energetics_df[energetics_df['track_id'].isin(top_3_systems_q90.index)].dropna()

# Adcionar coluna de EOF dominante
for track_id in top_3_systems_q10.index:
    dominant_eof = top_3_systems_q10.loc[track_id, 'dominant_eof']
    energetics_q10.loc[energetics_q10['track_id'] == track_id, 'dominant_eof'] = int(dominant_eof)

for track_id in top_3_systems_q90.index:
    dominant_eof = top_3_systems_q90.loc[track_id, 'dominant_eof']
    energetics_q90.loc[energetics_q90['track_id'] == track_id, 'dominant_eof'] = int(dominant_eof)

# Transformar eof em int
energetics_q10['dominant_eof'] = energetics_q10['dominant_eof'].astype(int)
energetics_q90['dominant_eof'] = energetics_q90['dominant_eof'].astype(int)

# Construir um índice com base no track_id e dominant_eof, usando (+) e (-) para diferenciar q10 de q90
energetics_q10['track_id_eof'] = energetics_q10['track_id'].astype(str) + '_EOF' + energetics_q10['dominant_eof'].astype(str) + '(+)'
energetics_q90['track_id_eof'] = energetics_q90['track_id'].astype(str) + '_EOF' + energetics_q90['dominant_eof'].astype(str) + '(-)'
energetics_q10 = energetics_q10.set_index('track_id_eof')
energetics_q90 = energetics_q90.set_index('track_id_eof')

# Remover colunas não relacionadas com energia
energetics_q10 = energetics_q10.drop(columns=['track_id', 'dominant_eof', 'lon vor', 'lat vor', 'vor42', 'region', 'period'])
energetics_q90 = energetics_q90.drop(columns=['track_id', 'dominant_eof', 'lon vor', 'lat vor', 'vor42', 'region', 'period'])

# Passar data para datetime
energetics_q10['date'] = pd.to_datetime(energetics_q10['date'])
energetics_q90['date'] = pd.to_datetime(energetics_q90['date'])

# Média tmeporal para cada track_id_eof
energetics_q10 = energetics_q10.groupby('track_id_eof').mean()
energetics_q90 = energetics_q90.groupby('track_id_eof').mean()

# Remover data
energetics_q10 = energetics_q10.drop(columns=['date'])
energetics_q90 = energetics_q90.drop(columns=['date'])

# Remover (finite diff.) das colunas
energetics_q10.columns = [col.replace(' (finite diff.)', '') for col in energetics_q10.columns]
energetics_q90.columns = [col.replace(' (finite diff.)', '') for col in energetics_q90.columns]

# # Reconstruir a energética para os sistemas com limiar 5
# reconstructed_energetics_q10 = reconstruct_energetics_from_pcs_and_eofs(top_3_systems_q10, eofs)
# reconstructed_energetics_q90 = reconstruct_energetics_from_pcs_and_eofs(top_3_systems_q90, eofs)

# # Exibir os resultados
# print("Energetics reconstruída para EOF_q10:")
# print(reconstructed_energetics_q10)

# print("Energetics reconstruída para EOF_q90:")
# print(reconstructed_energetics_q90)

# ######

def plot_LEC(reconstructed_energetics, figures_directory, suffix=''):
    from plot_LEC import _plotter

    # reconstructed_energetics['track_id'] = reconstructed_energetics['track_id'].astype(int)

    # # Juntar track_id e dominant_eof em uma coluna
    # reconstructed_energetics['dominant_eof'] = reconstructed_energetics['dominant_eof'].astype(int)
    # reconstructed_energetics['track_id_eof'] = reconstructed_energetics['track_id'].astype(str) + '_EOF' + reconstructed_energetics['dominant_eof'].astype(str)

    # # Adcionar sufixo
    # reconstructed_energetics['track_id_eof'] = reconstructed_energetics['track_id_eof'] + suffix

    # reconstructed_energetics = reconstructed_energetics.drop(columns=['track_id', 'dominant_eof'])
    # reconstructed_energetics = reconstructed_energetics.set_index('track_id_eof')

    # # Remover duplicatas
    # reconstructed_energetics = reconstructed_energetics[~reconstructed_energetics.index.duplicated(keep='first')]
    # reconstructed_energetics = reconstructed_energetics.sort_index()

    # Normalize data
    df_not_energy_periods = np.abs(
        reconstructed_energetics.drop(columns=["Az", "Ae", "Kz", "Ke"])
    )
    normalized_data_not_energy_periods = (
        df_not_energy_periods - df_not_energy_periods.min().mean()
    ) / (df_not_energy_periods.max().max() - df_not_energy_periods.min().min())
    normalized_data_not_energy_periods = normalized_data_not_energy_periods.clip(
        lower=1.5, upper=15
    )

    # Plot period means
    _plotter(reconstructed_energetics, normalized_data_not_energy_periods, figures_directory)

# Diretório para salvar as figuras
figures_directory = 'figures/LEC_reconstructed_top_PCs_q10_q90'

# Plotar as figuras
plot_LEC(energetics_q10, figures_directory, suffix='(+)')
plot_LEC(energetics_q90, figures_directory, suffix='(-)')