import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Caminhos para os arquivos
PATH = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic'
pcs_q10_path = f'{PATH}/csv_eofs_energetics_with_track/Total/pcs_with_dominant_eof_q10.csv'
pcs_q90_path = f'{PATH}/csv_eofs_energetics_with_track/Total/pcs_with_dominant_eof_q90.csv'

# Carregar os dados
pcs_q10 = pd.read_csv(pcs_q10_path)
pcs_q90 = pd.read_csv(pcs_q90_path)

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

# Definir as cores específicas para cada PC
pc_colors = {
    'PC1': '#5975A4',  # Azul
    'PC2': '#CC8963',  # Amarelo
    'PC3': '#B55D60',  # Vermelho
    'PC4': '#5F9E6E'   # Verde
}

output_dir = 'figures/eof_statistics_comparison_q10_q90'

# Criar uma figura com 2 linhas e 4 colunas
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Loop para gerar os boxplots para EOF_q90 (primeira linha)
for i, eof in enumerate([1, 2, 3, 4]):
    # Filtrar os dados para o EOF dominante igual a 1, 2, 3 ou 4 no q90
    filtered_df_q90 = merged_df[merged_df['dominant_eof_q90'] == eof]
    filtered_df_q90 = filtered_df_q90[['track_id', 'PC1', 'PC2', 'PC3', 'PC4']]
    filtered_df_q90 = filtered_df_q90.melt(id_vars=['track_id'], value_vars=['PC1', 'PC2', 'PC3', 'PC4'], var_name='PC', value_name='Value')
    
    sns.boxplot(data=filtered_df_q90, x='PC', y='Value', ax=axes[0, i], palette=pc_colors)
    axes[0, i].set_title(f'EOF_q90 = {eof}')
    axes[0, i].axhline(0, color='gray', linestyle='--')  # Linha horizontal no 0
    axes[0, i].set_xlabel('')
    axes[0, i].set_ylabel('')

# Loop para gerar os boxplots para EOF_q10 (segunda linha)
for i, eof in enumerate([1, 2, 3, 4]):
    # Filtrar os dados para o EOF dominante igual a 1, 2, 3 ou 4 no q10
    filtered_df_q10 = merged_df[merged_df['dominant_eof_q10'] == eof]
    filtered_df_q10 = filtered_df_q10[['track_id', 'PC1', 'PC2', 'PC3', 'PC4']]
    filtered_df_q10 = filtered_df_q10.melt(id_vars=['track_id'], value_vars=['PC1', 'PC2', 'PC3', 'PC4'], var_name='PC', value_name='Value')
    
    sns.boxplot(data=filtered_df_q10, x='PC', y='Value', ax=axes[1, i], palette=pc_colors)
    axes[1, i].set_title(f'EOF_q10 = {eof}')
    axes[1, i].axhline(0, color='gray', linestyle='--')  # Linha horizontal no 0
    axes[1, i].set_xlabel('')
    axes[1, i].set_ylabel('')

# Ajustar o layout da figura
plt.tight_layout()
plt.savefig(f'{output_dir}/boxplot_comparison_PCs_q10_q90.png', dpi=300)