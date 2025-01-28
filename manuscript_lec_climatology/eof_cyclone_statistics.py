import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Caminhos para os arquivos
track_path = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic/tracks_SAt_filtered/tracks_SAt_filtered_with_periods.csv'
pcs_path = '../../Programs_and_scripts/energetic_patterns_cyclones_south_atlantic/csv_eofs_energetics_with_track/Total/pcs_with_dominant_eof.csv'

# Criar diretório para salvar as figuras
output_dir = 'figures/eof_statistics'
os.makedirs(output_dir, exist_ok=True)

# Carregar os dados
tracks = pd.read_csv(track_path)
pcs = pd.read_csv(pcs_path)

# Converter a coluna 'date' para datetime
tracks['date'] = pd.to_datetime(tracks['date'])

# Unir os dados com base no track_id
merged_data = tracks.merge(pcs[['track_id', 'dominant_eof']], on='track_id')

# Restringir análise às primeiras 4 EOFs
merged_data = merged_data[merged_data['dominant_eof'].isin([1, 2, 3, 4])]

# Calcular estatísticas por ciclone
cyclone_stats = merged_data.groupby(['track_id', 'dominant_eof']).agg(
    max_intensity=('vor42', lambda x: x.max()), 
    mean_intensity=('vor42', lambda x: x.mean()), 
    duration=('date', lambda x: (x.max() - x.min()).total_seconds() / (3600 * 24))  # Duração em dias
).reset_index()

# Calcular o número de ciclones por EOF
cyclone_count = cyclone_stats.groupby('dominant_eof').size().reset_index(name='cyclone_count')

# Determinar a gênese: selecionar a primeira entrada de cada ciclone
genesis_data = merged_data.groupby('track_id').first().reset_index()

# Contar sistemas por região de gênese e EOF
region_counts = genesis_data.groupby(['dominant_eof', 'region']).size().reset_index(name='count')

# Calcular proporção por EOF
region_counts['proportion'] = region_counts.groupby('dominant_eof')['count'].transform(lambda x: x / x.sum())

# Converter proporções para porcentagens
region_counts['proportion'] *= 100

# Pivotar os dados para formato adequado para gráfico de barras empilhadas
proportion_pivot = region_counts.pivot(index='dominant_eof', columns='region', values='proportion').fillna(0)

# Configurar estilo para publicação científica
sns.set_context("notebook", font_scale=1.5)
sns.set_style("whitegrid")

# Paletas de cores para gráficos
palette = sns.color_palette("deep", n_colors=4)

# Boxplot da intensidade máxima por EOF
plt.figure(figsize=(10, 6))
sns.boxplot(data=cyclone_stats, x='dominant_eof', y='max_intensity', palette=palette)
plt.xlabel('EOF', fontsize=16)
plt.ylabel(r'Maximum $\zeta_{850}$  ($-10^{-5}$ s$^{-1}$)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'boxplot_max_intensity_per_eof.png'), dpi=300)

# Boxplot da intensidade média por EOF
plt.figure(figsize=(10, 6))
sns.boxplot(data=cyclone_stats, x='dominant_eof', y='mean_intensity', palette=palette)
plt.xlabel('EOF', fontsize=16)
plt.ylabel(r'Mean $\zeta_{850}$ ($-10^{-5}$ s$^{-1}$)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'boxplot_mean_intensity_per_eof.png'), dpi=300)

# Boxplot da duração média por EOF
plt.figure(figsize=(10, 6))
sns.boxplot(data=cyclone_stats, x='dominant_eof', y='duration', palette=palette)
plt.xlabel('EOF', fontsize=16)
plt.ylabel('Duration (days)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'boxplot_duration_per_eof.png'), dpi=300)

# Gráfico de barras do número de ciclones por EOF
plt.figure(figsize=(10, 6))
sns.barplot(data=cyclone_count, x='dominant_eof', y='cyclone_count', palette=palette)
plt.xlabel('EOF', fontsize=16)
plt.ylabel('Number of cyclones', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'barplot_cyclone_count_per_eof.png'), dpi=300)

# Gráfico de barras da proporção por região de gênese e EOF
proportion_pivot.plot(kind='bar', stacked=False, figsize=(12, 8), color=palette)
plt.xlabel('EOF', fontsize=16)
plt.ylabel('Proportion of Systems (%)', fontsize=16)  # Atualizar o rótulo do eixo Y
plt.legend(fontsize=12)
plt.xticks(rotation=0, fontsize=14)  # Ajustar rotação do eixo X
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'stacked_bar_genesis_proportion_per_eof.png'), dpi=300)
