import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


# Abrir dados das tracks com energética
energetics_path = f'{PATH}/tracks_SAt_filtered/tracks_SAt_filtered_with_energetics.csv'
energetics_df = pd.read_csv(energetics_path)

intense_systems = energetics_df[energetics_df['vor42'] > energetics_df['vor42'].quantile(0.9)]

# Selecionando os track_id dos sistemas intensos
intense_systems_ids = intense_systems['track_id']

# Filtrando os dados de merged_df para incluir apenas os track_ids dos sistemas intensos
intense_merged_df = merged_df[merged_df['track_id'].isin(intense_systems_ids)]


# Criando uma figura com duas subfiguras
fig, axes = plt.subplots(1, 2, figsize=(15, 8))

# Plotando todas as séries temporais na mesma figura
for pc in ['PC1', 'PC2', 'PC3', 'PC4']:
    axes[0].plot(intense_merged_df['track_id'], intense_merged_df[pc], label=pc)

axes[0].set_title('Séries Temporais de PC1 a PC4')
axes[0].set_xlabel('track_id')
axes[0].set_ylabel('Valor de PC')
axes[0].legend()

# Plotando os boxplots para cada PC
axes[1].boxplot([intense_merged_df[pc] for pc in ['PC1', 'PC2', 'PC3', 'PC4']])
axes[1].set_title('Boxplots de PC1 a PC4')
axes[1].set_xticklabels(['PC1', 'PC2', 'PC3', 'PC4'])
axes[1].set_ylabel('Valor de PC')

plt.tight_layout()
plt.show()


# Criando uma figura para o gráfico de barras empilhadas
fig, ax = plt.subplots(figsize=(15, 8))

# Preparando os dados para o gráfico de barras empilhadas
stacked_data = intense_merged_df[['PC1', 'PC2', 'PC3', 'PC4']]

# Plotando as barras empilhadas
stacked_data.plot(kind='bar', stacked=True, ax=ax, width=1, colormap='tab10')

ax.set_title('Gráfico de Barras Empilhadas para PC1 a PC4')
ax.set_xlabel('Data')
ax.set_ylabel('Valor de PC')
ax.legend(title='Componentes Principais')

plt.tight_layout()
plt.show()

# Definindo os percentis de intensidade para classificar os ciclones
percentiles = [0, 15, 35, 65, 85, 100]
intensity_labels = ['0-15', '15-35', '35-65', '65-85', '85-100']

# Calcular os percentis para 'vor42'
percentile_values = np.percentile(energetics_df['vor42'].dropna(), percentiles[1:-1])

# Categorizar os ciclones com base nos percentis de 'vor42'
energetics_df['intensity_category'] = pd.cut(energetics_df['vor42'], 
                                              bins=[-float('inf')] + list(percentile_values) + [float('inf')], 
                                              labels=intensity_labels)

# Extrair o ano a partir do track_id (primeiros 4 dígitos)
energetics_df['year_from_track'] = energetics_df['track_id'].astype(str).str[:4].astype(int)

# Definindo os intervalos de 5 anos, começando em 1980 até 2020
bins = list(range(1980, 2025, 5))  # Intervalos de 5 anos
labels = [f'{start}-{start+4}' for start in bins[:-1]]  # Criando os rótulos para os intervalos

# Adicionando a coluna de intervalo de 5 anos
energetics_df['five_years'] = pd.cut(energetics_df['year_from_track'], bins=bins, labels=labels, right=False)

# Contando o número de casos por intervalo de 5 anos e categoria de intensidade
intensity_counts = pd.crosstab(energetics_df['five_years'], energetics_df['intensity_category'])

# Criando o gráfico de barras empilhadas para a contagem de casos por intensidade a cada 5 anos
intensity_counts.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='tab10')

# Criando uma figura com subgráficos para cada categoria de intensidade
fig, axes = plt.subplots(1, 5, figsize=(20, 6), sharey=True)

# Para cada categoria de intensidade
for i, category in enumerate(intensity_labels):
    # Filtrando os dados para a categoria de intensidade
    category_data = energetics_df[energetics_df['intensity_category'] == category]
    
    # Contando o número de casos por intervalo de 5 anos
    category_counts = category_data['five_years'].value_counts().sort_index()

    # Plotando o gráfico de barras empilhadas para cada categoria de intensidade
    axes[i].bar(category_counts.index, category_counts.values, color='lightblue')
    axes[i].set_title(f'Intensidade: {category}')
    axes[i].set_xlabel('Intervalo de 5 Anos')
    axes[i].set_ylabel('Número de Ciclones')
    axes[i].tick_params(axis='x', rotation=45)

# Ajustando o layout para melhor visualização
plt.tight_layout()
plt.show()